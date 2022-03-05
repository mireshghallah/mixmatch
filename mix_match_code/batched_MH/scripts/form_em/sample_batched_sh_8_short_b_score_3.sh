


# python ../sample_batched.py \
# --max_iter 12 \
# --shuffle_positions \
# --temperature 1.0 \
# --out_path ../../output_samples/form_em \
# --alpha 100  \
# --beta 1 \
# --delta 30 \
# --gamma 300 \
# --data_name form_em_test_sh8 \
# --disc_name  frm_250 \
# --disc_dir  /home/user/dir.projects/sent_analysis/sent_anlys/clsf_train/form_em/models/checkpoint-250 \
# --data_path ../../data/form_em/test_sh_8.txt \
# --attr_path ../../data/form_em/test_sh_8.attr \

python ../sample_batched_form_em_len.py \
--max_iter 5 \
--shuffle_positions \
--temperature 1.0 \
--out_path ../../output_samples/form_em \
--alpha 140  \
--beta 1 \
--delta 30 \
--gamma 0 \
--theta 300 \
--data_name form_em_test_sh8_len_b_sc_r_inf \
--disc_name  frm_new \
--disc_dir  /home/user/dir.projects/sent_analysis/sent_anlys/clsf_train/form_em/models \
--data_path ../../data/form_em/test_sh_8_inf.txt \
--attr_path ../../data/form_em/test_sh_8_inf.attr \







