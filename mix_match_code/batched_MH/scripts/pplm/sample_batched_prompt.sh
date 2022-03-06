


python ./mix_match_code/batched_MH/scripts/sample_batched_prompt.py \
--max_iter 15 \
--max_len 12 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/pplm \
--alpha 40  \
--beta 1 \
--delta 0 \
--gamma 0 \
--data_name pplm_28_sentiment_test \
--disc_name  yelp_100 \
--disc_dir  /home/fmireshg/berglab.projects/sent_analysis/sent_anlys/clsf_train/yelp_cls_2/models/checkpoint-100 \
--src_path ./data/pplm/test.txt \
--attr_path ./data/pplm/test.attr \

