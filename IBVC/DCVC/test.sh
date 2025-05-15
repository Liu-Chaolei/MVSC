# ./test.sh
ROOT=./
export PYTHONPATH=$PYTHONPATH:$ROOT
CUDA_VISIBLE_DEVICES=3 python -u $ROOT/subnet/main.py --log loguvg.txt --testuvg --config config256.json \
#    --pretrain $ROOT/snapshot/256iter1453770.model