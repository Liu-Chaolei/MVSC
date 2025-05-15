# ./test.sh
ROOT=./
export PYTHONPATH=$PYTHONPATH:$ROOT
CUDA_VISIBLE_DEVICES=0 python -u $ROOT/subnet/main.py --log loguvg.txt --testuvg --config config256.json \
    --pretrain /data/ssd/liuchaolei/models/IBVC/snapshot256/iter1938360.model