export CPU_NUM=1
#CUDA_VISIBLE_DEVICES=3 gdb -ex r --args python train.py \
CUDA_VISIBLE_DEVICES=2  python train.py \
	--data_path baike  \
	--vocab_path baike/vocabulary_min5k.txt \
	--log_interval 1 \
	--para_print \
        --optim sgd \
	--embed_size 2 \
	--hidden_size 3 \
	--batch_size 2 \
	--detail \
	--dropout 0.0 \
	--num_steps 5 \
	--para_init \
	--learning_rate 0.5 \
	--train_path '/paddle/bilm-tf/./baike/train/sentence_file_19199.txt' \
	--test_path '/paddle/bilm-tf/./baike/train/sentence_file_19199.txt' \
	--para_load_dir '/paddle/bilm-tf/output/para-random/' \
	$@
