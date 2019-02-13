export CPU_NUM=1
#CUDA_VISIBLE_DEVICES=5  python  train.py \
CUDA_VISIBLE_DEVICES=0 python train.py \
	--data_path baike  \
	--vocab_path baike/vocabulary_min5k.txt \
	--dropout 0.0 \
        --local True \
        --use_gpu True \
	--learning_rate 0.2 \
	--train_path '/paddle/bilm-tf/./baike/train/sentence_file_*.txt' \
	--test_path '/paddle/bilm-tf/./baike/train/sentence_file_849.txt' \
        --para_load_dir '/paddle/bilm-tf/output/para-random3/' --para_print $@
