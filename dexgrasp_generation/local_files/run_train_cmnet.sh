cd ../
export CUDA_VISIBLE_DEVICES=0
python ./network/train.py --config-name cm_net_config_train --exp-dir ./runs/cm_net_train_batch128_2
