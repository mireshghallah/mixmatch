# python sample_batched.py \
# --max_iter 8 \
# --shuffle_positions \
# --temperature 1.0 \
# --out_path ./output_samples/yelp_imdb \
# --alpha 100  \
# --beta 1 \
# --delta 50 \
# --data_name yelp_li_test \
# --disc_name  yelp_100 \
# --disc_dir  /home/user/dir_projects/dir.bert_sample/sent_anlys/clsf_train/yelp_cls_2/models/checkpoint-100 \
# --data_path data/bias/test.txt \
# --attr_path data/bias/test.attr \



python ../sample_batched_topic_prompt.py \
--max_iter 15 \
--max_len 20 \
--shuffle_positions \
--temperature 1.0 \
--out_path ../../output_samples/topic \
--alpha 0  \
--beta 1 \
--delta 0 \
--gamma 0 \
--theta 0 \
--batch_size 20 \
--data_name test_2_full_energy  \
--disc_name  none \
--disc_dir  /home/user/dir.projects/sent_analysis/sent_anlys/clsf_train/yelp_cls_2/models/checkpoint-100 \
--src_path ../../data/topic/test_2.txt \
--attr_path ../../data/topic/test_2.attr \

