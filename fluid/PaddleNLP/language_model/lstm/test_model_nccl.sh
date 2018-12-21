#!/bin/bash
set -xe

source ./user.sh

source ./env.sh

export PADDLE_IS_LOCAL=0
export FLAGS_fraction_of_gpu_memory_to_use=0.15

echo $LD_LIBRARY_PATH

echo "run on trainer:" ${PADDLE_TRAINER_ID}

python  train.py \
--train_path='baike/train/sentence_file_*'  \
--test_path='baike/dev/sentence_file_*'  \
--vocab_path baike/vocabulary_min5k.txt \
--learning_rate 0.2 \
--use_gpu True \
--local False \
--shuffle True \
--update_method nccl2  > ${PADDLE_TRAINER_ID}.log  2>&1 &
