CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python  train.py \
--train_path='baike/train/sentence_file_*'  \
--test_path='baike/dev/sentence_file_*'  \
--vocab_path baike/vocabulary_min5k.txt \
--learning_rate 0.001 \
--use_gpu True $@
