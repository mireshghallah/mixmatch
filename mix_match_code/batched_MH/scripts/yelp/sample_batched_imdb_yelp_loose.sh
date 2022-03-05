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
# --data_path data/yelp/test_li.txt \
# --attr_path data/yelp/test_li.attr \



python sample_batched.py \
--max_iter 8 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/yelp_imdb \
--alpha 140  \
--beta 1 \
--delta 50 \
--data_name yelp_li_test \
--disc_name  yelp_fabriceyhc \
--disc_dir  fabriceyhc/bert-base-uncased-yelp_polarity \
--data_path data/yelp/test_li.txt \
--attr_path data/yelp/test_li.attr \


python sample_batched.py \
--max_iter 8 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/yelp_imdb \
--alpha 100  \
--beta 1 \
--delta 50 \
--data_name yelp_li_test \
--disc_name  yelp_fabriceyhc \
--disc_dir  fabriceyhc/bert-base-uncased-yelp_polarity \
--data_path data/yelp/test_li.txt \
--attr_path data/yelp/test_li.attr \


# python sample_batched.py \
# --max_iter 8 \
# --shuffle_positions \
# --temperature 1.0 \
# --out_path ./output_samples/yelp_imdb \
# --alpha 100  \
# --beta 1 \
# --delta 50 \
# --data_name yelp_li_test \
# --disc_name  yelp_300 \
# --disc_dir  /home/user/dir_projects/dir.bert_sample/sent_anlys/clsf_train/yelp_cls_2/models/checkpoint-300 \
# --data_path data/yelp/test_li.txt \
# --attr_path data/yelp/test_li.attr \


# python sample_batched.py \
# --max_iter 8 \
# --shuffle_positions \
# --temperature 1.0 \
# --out_path ./output_samples/yelp_imdb \
# --alpha 20  \
# --beta 1 \
# --delta 40 \
# --data_name yelp_li_test \
# --disc_name  yelp \
# --disc_dir  /home/user/dir_projects/dir.bert_sample/sent_anlys/clsf_train/yelp_cls_2/models/checkpoint-400 \
# --data_path data/yelp/test_li.txt \
# --attr_path data/yelp/test_li.attr \


# python sample_batched.py \
# --max_iter 8 \
# --shuffle_positions \
# --temperature 1.0 \
# --out_path ./output_samples/yelp_imdb \
# --alpha 50  \
# --beta 1 \
# --delta 40 \
# --data_name yelp_li_test \
# --disc_name  yelp \
# --disc_dir  /home/user/dir_projects/dir.bert_sample/sent_anlys/clsf_train/yelp_cls_2/models/checkpoint-400 \
# --data_path data/yelp/test_li.txt \
# --attr_path data/yelp/test_li.attr \

#--block \