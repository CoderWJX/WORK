""" Train a network."""
from typing import Optional
import logging
import argparse
import warnings

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

from os.path import isdir
from pathlib import Path

import torch
import time
from nequip.model import model_from_config
from nequip.utils import Config
from nequip.data import dataset_from_config, register_fields, AtomicData
from nequip.utils import load_file, instantiate, dtype_from_name
from nequip.utils.test import assert_AtomicData_equivariant
from nequip.utils.versions import check_code_version
# from nequip.utils._global_options import _set_global_options
from nequip.scripts._logger import set_up_script_logger
from nequip.scripts.deploy import load_deployed_model, R_MAX_KEY
from nequip.train import Trainer, Loss, Metrics
import os
import copy
from allegro.nn.qt_util import get_config, quantizer
import allegro.nn.quant.quant_qat_converter as quant_converter
import allegro.nn.quant.quantized_converter as quantized_converter

import contextlib
from tqdm.auto import tqdm
import munch
import e3nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

default_config = dict(
    root="./",
    run_name="NequIP",
    wandb=False,
    wandb_project="NequIP",
    model_builders=[
        "SimpleIrrepsConfig",
        "EnergyModel",
        "PerSpeciesRescale",
        "ForceOutput",
        "RescaleEnergyEtc",
    ],
    dataset_statistics_stride=1,
    default_dtype="float32",
    allow_tf32=False,  # TODO: until we understand equivar issues
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
    append=False,
    _jit_bailout_depth=2,  # avoid 20 iters of pain, see https://github.com/pytorch/pytorch/issues/52286
    # Quote from eelison in PyTorch slack:
    # https://pytorch.slack.com/archives/CDZD1FANA/p1644259272007529?thread_ts=1644064449.039479&cid=CDZD1FANA
    # > Right now the default behavior is to specialize twice on static shapes and then on dynamic shapes.
    # > To reduce warmup time you can do something like setFusionStrartegy({{FusionBehavior::DYNAMIC, 3}})
    # > ... Although we would wouldn't really expect to recompile a dynamic shape fusion in a model,
    # > provided broadcasting patterns remain fixed
    # We default to DYNAMIC alone because the number of edges is always dynamic,
    # even if the number of atoms is fixed:
    _jit_fusion_strategy=[("DYNAMIC", 3)],
)

ORIGINAL_DATASET_INDEX_KEY: str = "original_dataset_index"
register_fields(graph_fields=[ORIGINAL_DATASET_INDEX_KEY])


def save_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

def _set_global_options(config):
    """Configure global options of libraries like `torch` and `e3nn` based on `config`."""
    # Set TF32 support
    # See https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if torch.cuda.is_available():
        if torch.torch.backends.cuda.matmul.allow_tf32 and not config.allow_tf32:
            # it is enabled, and we dont want it to, so disable:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    if int(torch.__version__.split(".")[1]) >= 11:
        # PyTorch >= 1.11
        k = "_jit_fusion_strategy"
        torch.jit.set_fusion_strategy(config.get(k))
    else:
        # For avoiding 20 steps of painfully slow JIT recompilation
        # See https://github.com/pytorch/pytorch/issues/52286
        torch._C._jit_set_bailout_depth(config["_jit_bailout_depth"])

    if config.model_debug_mode:
        set_irreps_debug(enabled=True)
    torch.set_default_dtype(dtype_from_name(config.default_dtype))
    if config.grad_anomaly_mode:
        torch.autograd.set_detect_anomaly(True)

    e3nn.set_optimization_defaults(**config.get("e3nn_optimization_defaults", {}))

    # Register fields:
    instantiate(register_fields, all_args=config)

def print_para_num(model, name):
    total = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(name + 'Total parameter number: %d, Trainable parameter num: %d' %(total, trainable_num))

def qat_train(trainer, config):
    fp32_model = trainer.model
    print_para_num(fp32_model, "fp32_model")

    qt_model = fp32_model

    # Model and fused model should be equivalent.
    fp32_model.eval()

    # # Use training data for calibration.
    print("Training QAT Model...")
    qt_model.train()

    trainer = fresh_start(config)

    replaced_modules = quant_converter.find_modules_to_quantize(trainer.model, munch.munchify(config.quan))

    trainer.model = quant_converter.replace_module_by_names(trainer.model, replaced_modules)

    # print("Before QAT training...")
    # evaluate(trainer, config.train_dir, config)

    trainer.train()

    print("After QAT training...")

    evaluate_qat_model(trainer, config)


def evaluate(trainer, config, device="cuda"):
    # Use the model config, regardless of dataset config
    global_config = config.train_dir + "/config.yaml"
    global_config = Config.from_file(str(global_config), defaults=default_config)
    _set_global_options(global_config)
    check_code_version(global_config)
    del global_config
    trainer.model = trainer.model.to(device)

    print("loaded model from training session")
    model_r_max = config["r_max"]
    trainer.model.eval()

    if config["r_max"] != model_r_max:
        raise RuntimeError(
            f"Dataset config has r_max={config['r_max']}, but model has r_max={model_r_max}!"
        )

    # dataset_is_validation: bool = False
    try:
        # Try to get validation dataset
        dataset = dataset_from_config(config, prefix="evaluate_dataset")
        dataset_is_validation = True
    except KeyError:
        pass


    from nequip.data import Collater
    c = Collater.for_dataset(dataset, exclude_keys=[])

    # Determine the test set
    # this makes no sense if a dataset is given seperately
    test_idcs = torch.arange(dataset.len())
    logging.info(
        f"Using all frames from the specified test dataset, yielding a test set size of {len(test_idcs)} frames.",
    )

    test_idcs = torch.as_tensor(test_idcs, dtype=torch.long)
    test_idcs = test_idcs.tile((config.repeat,))
    do_metrics = True
    # Figure out what metrics we're actually computing
    if do_metrics:
        metrics_config = Config.from_file(str(config.metrics_config))
        metrics_components = metrics_config.get("metrics_components", None)
        # See trainer.py: init() and init_metrics()
        # Default to loss functions if no metrics specified:
        if metrics_components is None:
            loss, _ = instantiate(
                builder=Loss,
                prefix="loss",
                positional_args=dict(coeffs=metrics_config.loss_coeffs),
                all_args=metrics_config,
            )
            metrics_components = []
            for key, func in loss.funcs.items():
                params = {
                    "PerSpecies": type(func).__name__.startswith("PerSpecies"),
                }
                metrics_components.append((key, "mae", params))
                metrics_components.append((key, "rmse", params))

        metrics, _ = instantiate(
            builder=Metrics,
            prefix="metrics",
            positional_args=dict(components=metrics_components),
            all_args=metrics_config,
        )
        metrics.to(device=device)

    batch_i: int = 0
    batch_size: int = config.batch_size

    logging.info("Starting...")
    start = time.time()
    context_stack = contextlib.ExitStack()
    with contextlib.ExitStack() as context_stack:
        # "None" checks if in a TTY and disables if not
        prog = context_stack.enter_context(tqdm(total=len(test_idcs), disable=None))
        if do_metrics:
            display_bar = context_stack.enter_context(
                tqdm(
                    bar_format=""
                    if prog.disable  # prog.ncols doesn't exist if disabled
                    else ("{desc:." + str(prog.ncols) + "}"),
                    disable=None,
                )
            )
        output_type = None
        if output_type is not None:
            output = context_stack.enter_context(open(config.output, "w"))
        else:
            output = None

        while True:
            this_batch_test_indexes = test_idcs[
                                      batch_i * batch_size: (batch_i + 1) * batch_size]

            datas = [dataset[int(idex)] for idex in this_batch_test_indexes]
            if len(datas) == 0:
                break
            batch = c.collate(datas)
            batch = batch.to(device)

            out = trainer.model(AtomicData.to_AtomicDataDict(batch))

            with torch.no_grad():
                # Write output
                if output_type == "xyz":
                    # add test frame to the output:
                    out[ORIGINAL_DATASET_INDEX_KEY] = torch.LongTensor(
                        this_batch_test_indexes
                    )
                    # append to the file
                    ase.io.write(
                        output,
                        AtomicData.from_AtomicDataDict(out)
                            .to(device="cpu")
                            .to_ase(
                            type_mapper=dataset.type_mapper,
                            extra_fields=args.output_fields,
                        ),
                        format="extxyz",
                        append=True,
                            )

                # Accumulate metrics
                if do_metrics:
                    metrics(out, batch)
                    display_bar.set_description_str(
                        " | ".join(
                            f"{k} = {v:4.4f}"
                            for k, v in metrics.flatten_metrics(
                                metrics.current_result(),
                                type_names=dataset.type_mapper.type_names,
                            )[0].items()
                        )
                    )

            batch_i += 1
            prog.update(batch.num_graphs)

        prog.close()
        end = time.time()

        if do_metrics:
            display_bar.close()
        logging.info(f"\n--- Evaluation Time consumption: {(end - start):3f}s ---")

    if do_metrics:
        logging.info("\n--- Evaluation Final result: ---")
        logging.critical(
            "\n".join(
                f"{k:>20s} = {v:< 20f}"
                for k, v in metrics.flatten_metrics(
                    metrics.current_result(),
                    type_names=dataset.type_mapper.type_names,
                )[0].items()
            )
        )


def evaluate_qat_model(trainer, config):
    model = trainer.model

    # build a QAT model
    trainer.model.eval()
    replaced_modules = quant_converter.find_modules_to_quantize(trainer.model, munch.munchify(config.quan))
    trainer.model = quant_converter.replace_module_by_names(trainer.model, replaced_modules)

    qat_file_path = config.train_dir + "/best_model.pth"
    load_state_dict = torch.load(qat_file_path)
    model.load_state_dict(load_state_dict)

    trainer.model.eval()
    print("QAT model inference...")

    evaluate(trainer, config)
    trainer.save()

def evaluate_quantized_model(trainer, config):
    model = trainer.model
    # build a QAT model
    trainer.model.eval()
    replaced_modules = quant_converter.find_modules_to_quantize(trainer.model, munch.munchify(config.quan))
    trainer.model = quant_converter.replace_module_by_names(trainer.model, replaced_modules)

    qat_file_path = config.train_dir + "/best_model.pth"
    load_state_dict = torch.load(qat_file_path)
    model.load_state_dict(load_state_dict)

    # convert QAT model into quantized model
    modules_to_inference = quantized_converter.find_modules_to_convert(model)
    trainer.model = quantized_converter.replace_module_to_inference(model, modules_to_inference)
    trainer.model.eval()
    print("Quantized model inference...")
    evaluate(trainer, config)
    trainer.save()


def main(args=None, running_as_script: bool = True):
    config = parse_command_line(args)
    print(config)
    if running_as_script:
        set_up_script_logger(config.get("log", None), config.verbose)

    found_restart_file = isdir(f"{config.root}/{config.run_name}")
    if found_restart_file and not config.append:
        raise RuntimeError(
            f"Training instance exists at {config.root}/{config.run_name}; "
            "either set append to True or use a different root or runname"
        )
    trainer = fresh_start(config)

    if config.taskname == "train":
        trainer.train()
    elif config.taskname == "qat":
        qat_train(trainer, config)
    elif config.taskname == "qat_inference":
        evaluate_qat_model(trainer, config)
    elif config.taskname == "inference":
        evaluate_quantized_model(trainer, config)
    else:
        raise RuntimeError(f"Unspecified task type: %s" % config.taskname)
    return


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(description="Train a NequIP model.")
    parser.add_argument("config", help="configuration file")
    parser.add_argument(
        "--equivariance-test",
        help="test the model's equivariance before training on n (default 1) random frames from the dataset",
        const=1,
        type=int,
        nargs="?",
    )
    parser.add_argument(
        "--model-debug-mode",
        help="enable model debug mode, which can sometimes give much more useful error messages at the cost of some speed. Do not use for production training!",
        action="store_true",
    )
    parser.add_argument(
        "--grad-anomaly-mode",
        help="enable PyTorch autograd anomaly mode to debug NaN gradients. Do not use for production training!",
        action="store_true",
    )
    parser.add_argument(
        "--log",
        help="log file to store all the screen logging",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--test_indexes",
        help="Path to a file containing the indexes in the dataset that make up the test set. If omitted, all data frames *not* used as training or validation data in the training session `train_dir` will be used.",
        type=Path,
        default=None,
    )

    parser.add_argument("--local_rank", default=-1)

    args = parser.parse_args(args=args)

    config = Config.from_file(args.config, defaults=default_config)
    for flag in ("model_debug_mode", "equivariance_test", "grad_anomaly_mode"):
        config[flag] = getattr(args, flag) or config[flag]

    return config


def fresh_start(config):
    # we use add_to_config cause it's a fresh start and need to record it
    check_code_version(config, add_to_config=True)
    _set_global_options(config)

    # = Make the trainer =
    if config.wandb:
        import wandb  # noqa: F401
        from nequip.train.trainer_wandb import TrainerWandB

        # download parameters from wandb in case of sweeping
        from nequip.utils.wandb import init_n_update

        config = init_n_update(config)

        trainer = TrainerWandB(model=None, **dict(config))
    else:
        from nequip.train.trainer import Trainer

        trainer = Trainer(model=None, **dict(config))

    # what is this
    # to update wandb data?
    config.update(trainer.params)

    # = Load the dataset =
    dataset = dataset_from_config(config, prefix="dataset")
    logging.info(f"Successfully loaded the data set of type {dataset}...")
    try:
        validation_dataset = dataset_from_config(config, prefix="validation_dataset")
        logging.info(
            f"Successfully loaded the validation data set of type {validation_dataset}..."
        )
    except KeyError:
        # It couldn't be found
        validation_dataset = None

    # = Train/test split =
    trainer.set_dataset(dataset, validation_dataset)

    final_model = load_model(config)
    logging.info("Successfully load the pretrained model...")

    # by doing this here we check also any keys custom builders may have added
    _check_old_keys(config)

    # Equivar test
    if config.equivariance_test > 0:
        n_train: int = len(trainer.dataset_train)
        assert config.equivariance_test <= n_train
        final_model.eval()
        indexes = torch.randperm(n_train)[: config.equivariance_test]
        errstr = assert_AtomicData_equivariant(
            final_model, [trainer.dataset_train[i] for i in indexes]
        )
        final_model.train()
        logging.info(
            "Equivariance test passed; equivariance errors:\n"
            "   Errors are in real units, where relevant.\n"
            "   Please note that the large scale of the typical\n"
            "   shifts to the (atomic) energy can cause\n"
            "   catastrophic cancellation and give incorrectly\n"
            "   the equivariance error as zero for those fields.\n"
            f"{errstr}"
        )
        del errstr, indexes, n_train

    # Set the trainer
    trainer.model = final_model

    # Store any updated config information in the trainer
    trainer.update_kwargs(config)

    return trainer


def load_model(configs):
    # Parse the args
    if configs.base_train_dir:
        if 'dataset_config' not in configs or configs.dataset_config is None:
            configs.dataset_config = configs.base_train_dir + "/config.yaml"
        if 'metrics_config' not in configs or configs.metrics_config is None:
            configs.metrics_config = configs.base_train_dir + "/config.yaml"
        if 'base_model_file' not in configs or configs.base_model_file is None:
            configs.base_model_file = configs.base_train_dir + "/best_model.pth"

    # update
    if configs.metrics_config == "None":
        configs.metrics_config = None
    elif configs.metrics_config is not None:
        configs.metrics_config = Path(configs.metrics_config)
    do_metrics = configs.metrics_config is not None
    # validate
    if configs.dataset_config is None:
        raise ValueError("--dataset-config or --train-dir must be provided")
    if configs.metrics_config is None and configs.output is None:
        raise ValueError(
            "Nothing to do! Must provide at least one of --metrics-config, --train-dir (to use training config for metrics), or --output"
        )
    if configs.base_model_file is None:
        raise ValueError("--model or --train-dir must be provided")

    configs.output_fields = []

    if configs.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(configs.device)

    logging.info(f"Using device: {device}")
    if device.type == "cuda":
        logging.info(
            "WARNING: please note that models running on CUDA are usually nondeterministc and that this manifests in the final test errors; for a _more_ deterministic result, please use `--device cpu`",
        )

    # Load model:
    logging.info("Loading model... ")

    try:
        model, metadata = load_deployed_model(
            configs.base_model_file,
            device=device,
            set_global_options=True,  # don't warn that setting
        )
        logging.info("loaded deployed model.")
        # the global settings for a deployed model are set by
        # set_global_options in the call to load_deployed_model
        # above
        loaded_deployed_model = True
    except ValueError:  # its not a deployed model
        loaded_deployed_model = False
    # we don't do this in the `except:` block to avoid "during handing of this exception another exception"
    # chains if there is an issue loading the training session model. This makes the error messages more
    # comprehensible:
    if not loaded_deployed_model:
        # Use the model config, regardless of dataset config
        global_config = configs.base_train_dir + "/config.yaml"
        global_config = Config.from_file(str(global_config), defaults=default_config)
        _set_global_options(global_config)
        check_code_version(global_config)
        del global_config

        # load a training session model
        model, model_config = Trainer.load_model_from_training_session(
            traindir=configs.base_train_dir, model_name="best_model.pth"
        )
        model = model.to(device)
        logging.info("loaded model from training session")

        return model


def restart(config):
    # load the dictionary
    restart_file = f"{config.root}/{config.run_name}/trainer.pth"
    dictionary = load_file(
        supported_formats=dict(torch=["pt", "pth"]),
        filename=restart_file,
        enforced_format="torch",
    )

    # note, "trainer.pth"/dictionary also store code versions,
    # which will not be stored in config and thus not checked here
    check_code_version(config)

    # recursive loop, if same type but different value
    # raise error

    config = Config(dictionary, exclude_keys=["state_dict", "progress"])

    # dtype, etc.
    _set_global_options(config)

    # note, the from_dict method will check whether the code version
    # in trainer.pth is consistent and issue warnings
    if config.wandb:
        from nequip.train.trainer_wandb import TrainerWandB
        from nequip.utils.wandb import resume

        resume(config)
        trainer = TrainerWandB.from_dict(dictionary)
    else:
        from nequip.train.trainer import Trainer

        trainer = Trainer.from_dict(dictionary)

    # = Load the dataset =
    dataset = dataset_from_config(config, prefix="dataset")
    logging.info(f"Successfully re-loaded the data set of type {dataset}...")
    try:
        validation_dataset = dataset_from_config(config, prefix="validation_dataset")
        logging.info(
            f"Successfully re-loaded the validation data set of type {validation_dataset}..."
        )
    except KeyError:
        # It couldn't be found
        validation_dataset = None
    trainer.set_dataset(dataset, validation_dataset)

    return trainer


def _check_old_keys(config) -> None:
    """check ``config`` for old/depricated keys and emit corresponding errors/warnings"""
    # compile_model
    k = "compile_model"
    if k in config:
        if config[k]:
            raise ValueError("the `compile_model` option has been removed")
        else:
            warnings.warn("the `compile_model` option has been removed")


if __name__ == "__main__":
    main(running_as_script=True)
