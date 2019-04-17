python correct_text.py --input_train_path ./data/AESW/input.aesw2016\(v1.2\)_train.txt  \
                       --target_train_path ./data/AESW/target.aesw2016\(v1.2\)_train.txt  \
                       --input_dev_path ./data/AESW/input.aesw2016\(v1.2\)_dev.txt \
                       --target_dev_path ./data/AESW/target.aesw2016\(v1.2\)_dev.txt \
                       --config SentencePairConfig \
                       --data_reader_type SentencePairReader \
                       --model_path ./sentence_pair_model
