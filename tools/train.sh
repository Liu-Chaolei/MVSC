# ./tools/train.sh
ROOT=./
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir $ROOT/snapshot
CUDA_VISIBLE_DEVICES=4 python -u $ROOT src/train.py -c $ROOT/configs/config.yaml
# CUDA_VISIBLE_DEVICES=4 python train.py -c config.yaml