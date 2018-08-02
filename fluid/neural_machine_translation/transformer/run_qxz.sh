export PADDLE_IS_LOCAL=1
export GLOG_v=0
export CUDA_VISIBLE_DEVICES=4
export GLOG_logtostderr=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
python -u train_qxz.py --src_vocab_fpath data/vocab_all.bpe.32000 --trg_vocab_fpath data/vocab_all.bpe.32000 --special_token '<s>' '<e>' '<unk>' --train_file_pattern data/train.tok.clean.bpe.32000.en-de --use_token_batch True --batch_size 2048 --sort_type pool --pool_size 102400 --shuffle True --sync True --shuffle_batch False weight_sharing True warmup_steps 16000
