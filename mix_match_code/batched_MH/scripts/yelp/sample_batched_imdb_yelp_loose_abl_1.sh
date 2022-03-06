python ./mix_match_code/batched_MH/scripts/sample_batched.py \
--max_iter 8 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/yelp \
--alpha 1  \
--beta 0 \
--delta 1 \
--data_name yelp_li_test \
--disc_name  yelp_100 \
--disc_dir  /home/fmireshg/berglab.projects/sent_analysis/sent_anlys/clsf_train/yelp_cls_2/models/checkpoint-100 \
--data_path ./data/yelp/test_li.txt \
--attr_path ./data/yelp/test_li.attr \

