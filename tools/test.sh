# ./tools/test.sh
ROOT=./
export PYTHONPATH=$PYTHONPATH:$ROOT
CUDA_VISIBLE_DEVICES=4 python -u $ROOT src/test.py -c $ROOT/configs/config.yaml
# CUDA_VISIBLE_DEVICES=4 python train.py -c config.yaml