CUDA_VISIBLE_DEVICES=2,3 python  -m pdb train.py --data_path baike  --vocab_path baike/vocabulary_min5k.txt --use_gpu True $@
