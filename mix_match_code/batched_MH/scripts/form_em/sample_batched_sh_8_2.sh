

python ../sample_batched.py \
--max_iter 12 \
--shuffle_positions \
--temperature 1.0 \
--out_path ../../output_samples/form_em \
--alpha 100  \
--beta 1 \
--delta 50 \
--gamma 300 \
--data_name form_em_test_sh8 \
--disc_name  frm_150 \
--disc_dir  /home/user/dir.projects/sent_analysis/sent_anlys/clsf_train/form_em.bckp/models/checkpoint-150 \
--data_path ../../data/form_em/test_sh_8.txt \
--attr_path ../../data/form_em/test_sh_8.attr \

python ../sample_batched.py \
--max_iter 12 \
--shuffle_positions \
--temperature 1.0 \
--out_path ../../output_samples/form_em \
--alpha 100  \
--beta 1 \
--delta 15 \
--gamma 300 \
--data_name form_em_test_sh8 \
--disc_name  frm_250 \
--disc_dir  /home/user/dir.projects/sent_analysis/sent_anlys/clsf_train/form_e.bckp/models/checkpoint-250 \
--data_path ../../data/form_em/test_sh_8.txt \
--attr_path ../../data/form_em/test_sh_8.attr \




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
# --data_path data/form_em/test.txt \
# --attr_path data/form_em/test.attr \


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
# --data_path data/form_em/test.txt \
# --attr_path data/form_em/test.attr \


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
# --data_path data/form_em/test.txt \
# --attr_path data/form_em/test.attr \

#--block \