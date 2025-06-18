now=$(date "+%Y_%m_%d_%H_%M_%S")
task_type="qat"
LR=0.002
EPOCH=100
#CUDA_VISIBLE_DEVICES=0 nohup /mnt/hwl/anaconda3/envs/zyj/bin/python train_qat.py /mnt/yujie.zeng/project_2023/allegro_models/Light-Allegro-3Layer-6rmax_qat_bs16/sin_qat.yaml  > ${now}_qat.log 2>&1 &

#sleep 10s

now=$(date "+%Y_%m_%d_%H_%M_%S")
CUDA_VISIBLE_DEVICES=0 nohup /mnt/hwl/anaconda3/envs/zyj/bin/python train_qat.py /mnt/yujie.zeng/project_2023/allegro_models/Light-Allegro-3Layer-6rmax_qat_bs8_epoch50/sin_qat.yaml  > ${now}_qat.log 2>&1 &
