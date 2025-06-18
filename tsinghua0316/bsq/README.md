# bsq


## 0 目录结构

```
- bsq/
  - datasets/
    - ncf/  
     + ml-1m/    #ml-1m数据集
  + extensions/   #扩展包
  + scripts/    #脚本
  + src/bsq/  #量化工具包
  - experiments/
    - resnet/
    - ncf/
    - bert/
  setup.py
  README.md
```

## 1 环境搭建

```shell_script
conda create -n bsq-env python=3.8
conda activate bsq-env
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install scikit-learn pyyaml munch
pip install transformers datasets #huggingface开发的transformers模块以及datasets数据加载模块
cp -r bsq ./
cd bsq
pip install ./extensions/bsq-ext/dist/bsq_ext-0.0.0-cp38-cp38-linux_x86_64.whl
pip install -e .
```

## 2 resnet50 实验

```shell_script
cd experiments/resnet
```

### 2.1resnet50 baseline 

```shell_script
python main.py config_files/resnet50_imagenet_baseline.yaml
```

&emsp;&emsp;通过设置yaml文件中的output_dir参数，可以指定模型训练的日志以及checkpoint的保存位置。模型训练结果将保存在output/{\$name}_{\$datetime}/{\$name}_best.pth。设置dataloader可以设置不同的数据集。

### 2.2 模型量化感知训练

```shell_script
python main.py config_files/resnet50_imagenet_a8w8.yaml
```

&emsp;&emsp;指定yaml文件中的pretrained_path参数为baseline模型的路径，确保task_name参数为'quant'，设定量化参数，设置output_dir参数。

### 2.3 将模型转成inference模型，并inference

```shell_script
python main.py config_files/resnet50_imagenet_a8w8_inference.yaml
```

&emsp;&emsp;设置task_name为'inference'，指定quant_path为量化感知训练的模型路径。


## 3 ncf训练


```shell_script
cd experiments/ncf
```

### 3.1 ncf baseline

```
python main.py config_files/mlp32_baseline.yaml
python main.py config_files/gmf32_baseline.yaml

# 将前两次训练得到的模型结合在一起然后funtuning
python main.py config_files/ncf32_baseline.yaml
```

### 3.2 ncf量化

```shell_script
python main.py config_files/ncf32_a8w8.yaml
```

### 3.3 ncf推理
```shell_script
python main.py config_files/ncf32_a8w8_inference.yaml
```
&emsp;$emsp;将于out目录生成int8模型，验证集第一个batch的各层激活输出

## 4 bert训练

```shell_script
cd experiments/bert
```

### 4.1 bert baseline

```shell_script
python run_qa.py \
  --model_name_or_path bert-large-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./out/debug_squad/
```

### 4.2 bert量化

```shell_script
python run_qa_quant.py \
  --model_name_or_path ./out/debug_squad/checkpoint-2500/ \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./out/debug_squad_quant/
```

### 4.3 bert 推理

```shell_script
python run_qa_inference.py \
  --model_name_or_path ./out/squad_quant/ \
  --dataset_name squad \
  --do_eval \
  --per_device_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./out/squad_inference/
```
&emsp;$emsp;将于out目录生成int8模型，验证集第一个batch的各层激活输出

## 5 实验结果

**resnet50**

|                   | top-1 | top-5 |
| ----------------- | ----- | ----- |
| resnet50-baseline | 75.79 | 92.75 |
| resnet50-quant    | 76.81 | 93.26 |

**ncf**

|              | HR@10  | HDCG@10 |
| ------------ | ------ | ------- |
| mlp          | 69.205 | 41.240  |
| gmf          | 70.894 | 42.816  |
| ncf-baseline | 72.003 | 43.941  |
| ncf-quant    | 72.036 | 43.927  |

**bert-large-uncased**

|          | f1    | exact_match |
| -------- | ----- | ----------- |
| baseline | 90.52 | 83.64       |
| quant    | 90.37 | 83.13       |

