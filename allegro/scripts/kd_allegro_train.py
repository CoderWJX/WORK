import argparse
import warnings
import numpy as np
import torch
import random
import contextlib

from nequip.utils._global_options import _set_global_options
from nequip.utils import Config
from nequip.data import dataset_from_config, register_fields, AtomicData

from nequip.utils.versions import check_code_version

from nequip.train._key import ABBREV, LOSS_KEY, TRAIN, VALIDATION
from nequip.scripts.train import fresh_start, restart

ORIGINAL_DATASET_INDEX_KEY: str = "original_dataset_index"
register_fields(graph_fields=[ORIGINAL_DATASET_INDEX_KEY])

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    print("seed is set as: ", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # dgl.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _check_old_keys(config) -> None:
    """check ``config`` for old/depricated keys and emit corresponding errors/warnings"""
    # compile_model
    k = "compile_model"
    if k in config:
        if config[k]:
            raise ValueError("the `compile_model` option has been removed")
        else:
            warnings.warn("the `compile_model` option has been removed")


def teacher_batch_step(trainer, data):
    data = data.to(trainer.torch_device)
    data = AtomicData.to_AtomicDataDict(data)

    data_unscaled = data
    for layer in trainer.rescale_layers:
        data_unscaled = layer.unscale(data_unscaled)

    # Run model
    # We make a shallow copy of the input dict in case the model modifies it
    input_data = {
        k: v
        for k, v in data_unscaled.items()
        if k not in trainer._remove_from_model_input
    }
    out = trainer.model(input_data)
    del input_data

    return [out, data_unscaled]


def teacher_train_epoch(trainer, config, device="cuda"):
    # Use the model config, regardless of dataset config
    # global_config = config.train_dir + "/config.yaml"
    # global_config = Config.from_file(str(global_config))
    # _set_global_options(global_config)
    # check_code_version(global_config)
    # del global_config
    trainer.model = trainer.model.to(device)

    teacher_file_path = "/Light-Allegro-3Layer-rmax6.0/Light-Allegro-3layer-Rmax6.0.pth"
    load_state_dict = torch.load(teacher_file_path)
    trainer.model.load_state_dict(load_state_dict)

    print("loaded model from training session")
    # model_r_max = config["r_max"]
    trainer.model.eval()

    dataloaders = {TRAIN: trainer.dl_train, VALIDATION: trainer.dl_val}
    categories = [TRAIN, VALIDATION] if trainer.iepoch >= 0 else [VALIDATION]
    dataloaders = [
        dataloaders[c] for c in categories
    ]  # get the right dataloaders for the catagories we actually run
    trainer.metrics_dict = {}
    trainer.loss_dict = {}

    for category, dataset in zip(categories, dataloaders):
        if category == VALIDATION and trainer.use_ema:
            cm = trainer.ema.average_parameters()
        else:
            cm = contextlib.nullcontext()

        with cm:
            trainer.reset_metrics()
            trainer.n_batches = len(dataset)
            t_res_full = []
            for trainer.ibatch, batch in enumerate(dataset):

                t_res_batch = teacher_batch_step(
                    data=batch,
                    validation=(category == VALIDATION),
                )
                trainer.end_of_batch_log(batch_type=category)
                for callback in trainer._end_of_batch_callbacks:
                    callback(trainer)
                t_res_full.append(t_res_batch)


            # if category == TRAIN:
            #     for callback in trainer._end_of_train_callbacks:
            #         callback(trainer)
        return t_res_full


def train_teacher(config):

    trainer = fresh_start(config)
    # state_dict = load_file()
    # trainer.model.load_state_dict(state_dict)
    out = teacher_train_epoch(trainer, config)
    return out

def student_batch_step(data, teacher_out):
    return

def train_student(config, teacher_out):
    trainer = fresh_start(config)
    # Use the model config, regardless of dataset config

    # global_config = config.train_dir + "/config.yaml"
    # global_config = Config.from_file(str(global_config))
    # _set_global_options(global_config)
    # check_code_version(global_config)
    # del global_config
    trainer.model = trainer.model.to(device)

    student_file_path = config.train_dir + "/best_model.pth"
    load_state_dict = torch.load(student_file_path)
    trainer.model.load_state_dict(load_state_dict)

    print("loaded model from training session")
    model_r_max = config["r_max"]
    trainer.model.eval()

    dataloaders = {TRAIN: trainer.dl_train, VALIDATION: trainer.dl_val}
    categories = [TRAIN, VALIDATION] if trainer.iepoch >= 0 else [VALIDATION]
    dataloaders = [
        dataloaders[c] for c in categories
    ]  # get the right dataloaders for the catagories we actually run
    trainer.metrics_dict = {}
    trainer.loss_dict = {}

    for category, dataset in zip(categories, dataloaders):
        if category == VALIDATION and trainer.use_ema:
            cm = trainer.ema.average_parameters()
        else:
            cm = contextlib.nullcontext()

        with cm:
            trainer.reset_metrics()
            trainer.n_batches = len(dataset)
            s_res_full = []
            for trainer.ibatch, batch in enumerate(dataset):
                s_res_batch = student_batch_step(batch, teacher_out[trainer.ibatch])
                trainer.end_of_batch_log(batch_type=category)
                for callback in trainer._end_of_batch_callbacks:
                    callback(trainer)
                s_res_full.append(s_res_batch)


def main(out_t=None):

    # model = Model(param).to(device)

    criterion_l = torch.nn.NLLLoss()
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    # evaluator = get_evaluator(param["dataset"])

    if param['distill_mode'] == 0:
        # optimizer = optim.Adam(model.parameters(), lr=float(1e-2), weight_decay=float(param["weight_decay"]))
        out, test_acc, test_val, test_best = train_teacher(teacher_config)
        return out, test_acc, test_val, test_best

        # check_writable(out_t_dir, overwrite=True)
        # np.savez(out_t_dir + "out", out.detach().cpu().numpy())

    else:
        # check_writable(output_dir, overwrite=True)
        # out_t = load_out_t(out_t_dir)
        # out_t = out_t.to(device)
        # optimizer = optim.Adam(model.parameters(), lr=float(param["learning_rate"]), weight_decay=float(param["weight_decay"]))
        test_acc, test_val, test_best = train_student(student_config, out_t)
        return test_acc, test_val, test_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--teacher", type=str, default="GCN", help="Teacher model")
    parser.add_argument("--student", type=str, default="MLP", help="Student model")
    parser.add_argument("--num_heads", type=int, default=4)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout_t", type=float, default=0.5)
    parser.add_argument("--dropout_s", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--lamb", type=float,default=0.1)

    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--max_epoch", type=int, default=500)

    parser.add_argument("--distill_mode", type=int, default=0, help="0: teacher; 1: student;")
    parser.add_argument("--exp_setting", type=str, default="tran", help="[tran, ind]")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--split_rate", type=float, default=0.2)
    parser.add_argument("--save_mode", type=int, default=1)
    parser.add_argument("--data_mode", type=int, default=1)
    parser.add_argument("--ablation_mode", type=int, default=0, help="0: FF-G2M; 1: valinna MLPs; 2: GLNN; 3: LFD; 3: HFD")
    parser.add_argument("--teacher_config", help="Teacher Allegro configuration file")
    parser.add_argument("--student_config", help="Teacher Allegro configuration file")


    args = parser.parse_args()
    param = args.__dict__
    # param.update(nni.get_next_parameter())
    teacher_config = Config.from_file(args.teacher_config)
    student_config = Config.from_file(args.student_config)


    print(param)

    # g, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
    # if args.exp_setting == "tran":
    #     indices = (idx_train, idx_val, idx_test)
    # elif args.exp_setting == "ind":
    #     indices = graph_split(idx_train, idx_val, idx_test, args.split_rate, args.seed)
    #
    # feats = g.ndata["feat"].to(device)
    # labels = labels.to(device)
    # param['feat_dim'] = g.ndata["feat"].shape[1]
    # param['label_dim'] = labels.int().max().item() + 1


    set_seed(param['seed'])
    param["distill_mode"] = 0
    out, _, test_teacher, _ = main()
    param["distill_mode"] = 1
    test_acc, test_val, test_best = main(out)
    # nni.report_final_result(test_val)