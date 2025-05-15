# ./run.sh
ROOT=./
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir $ROOT/snapshot2048
CUDA_VISIBLE_DEVICES=7 python -u $ROOT/subnet/main.py --log log.txt --config $ROOT/config2048.json \
#  --pretrain $ROOT/snapshot/iter1292240.model
