CUDA_VISIBLE_DEVICES=0,1,2,3 python  train.py --data_path baike  --vocab_path baike/vocabulary_min5k.txt --use_gpu True $@
