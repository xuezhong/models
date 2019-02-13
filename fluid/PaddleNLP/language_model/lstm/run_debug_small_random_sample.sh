export CPU_NUM=1
#CUDA_VISIBLE_DEVICES=5  python  train.py \
CUDA_VISIBLE_DEVICES=6 python train.py \
	--data_path baike  \
	--vocab_path baike/vocabulary_min5k.txt \
	--log_interval 1 \
	--para_print \
	--embed_size 2 \
	--hidden_size 3 \
	--batch_size 2 \
	--detail \
	--dropout 0.0 \
	--num_steps 5 \
	--para_init \
        --local True \
	--sample_softmax \
        --use_gpu True \
	--learning_rate 0.5 \
	--train_path '/paddle/bilm-tf/./baike/train/sentence_file_849.txt' \
	--test_path '/paddle/bilm-tf/./baike/train/sentence_file_849.txt' \
	--para_load_dir '/paddle/bilm-tf/output/para-random2/' \
	$@
