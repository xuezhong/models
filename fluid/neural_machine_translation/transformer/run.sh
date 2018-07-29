export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
export CUDA_VISIBLE_DEVICES=2
export CPU_NUM=1
python -m pdb train.py   --src_vocab_fpath data/vocab_all.bpe.32000   --trg_vocab_fpath data/vocab_all.bpe.32000   --special_token '<s>' '</s>' '<unk>'   --train_file_pattern data/train.tok.clean.bpe.32000.en-de   --use_token_batch True   --batch_size 1024   --sort_type pool   --pool_size 200000 --device CPU
