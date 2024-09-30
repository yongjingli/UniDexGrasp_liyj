cd ../
export CUDA_VISIBLE_DEVICES=1
python ./network/train.py --config-name ipdf_config_train --exp-dir ./runs/ipdf_train_batch128
