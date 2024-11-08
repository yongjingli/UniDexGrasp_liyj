#nohup sh run_train_glow_joint.sh > output.txt 2>&1 &

cd ../
export CUDA_VISIBLE_DEVICES=3
python ./network/train.py --config-name glow_joint_config_train.yaml --exp-dir ./runs/exp_joint_train/glow_train_batch128
