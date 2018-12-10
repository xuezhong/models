CUDA_VISIBLE_DEVICES=5,6 python -m pdb train.py \
	--data_path baike  \
	--vocab_path baike/vocabulary_min5k.txt \
	--use_gpu True \
	--log_interval 1 \
	--para_print \
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
	--para_load_dir '/paddle/bilm-tf/output/para-small/' \
	$@
