cd ../
export CUDA_VISIBLE_DEVICES=2
python ./network/train.py --config-name glow_config_train --exp-dir ./runs/glow_train_batch128q
