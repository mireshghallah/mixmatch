



python ./mix_match_code/batched_MH/scripts/sample_batched_bias_boost.py \
--max_iter 8 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/bias \
--alpha 100  \
--beta 1 \
--delta 50 \
--gamma 0 \
--theta 100 \
--data_name bias_test_bi_boost \
--disc_name  bias_1050 \
--disc_dir  /home/fmireshg/berglab.projects/sent_analysis/sent_anlys/clsf_train/bias/models/checkpoint-1050 \
--data_path ./data/bias/test_bi.txt \
--attr_path ./data/bias/test_bi.attr \

