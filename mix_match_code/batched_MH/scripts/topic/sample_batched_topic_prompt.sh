
python ./mix_match_code/batched_MH/scripts/sample_batched_topic_prompt.py \
--max_iter 15 \
--max_len 20 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/topic \
--alpha 0  \
--beta 1 \
--delta 0 \
--gamma 0 \
--theta 0 \
--batch_size 20 \
--data_name test_2_full_energy  \
--disc_name  none \
--disc_dir  /home/fmireshg/berglab.projects/sent_analysis/sent_anlys/clsf_train/yelp_cls_2/models/checkpoint-100 \
--src_path ./data/topic/test_2.txt \
--attr_path ./data/topic/test_2.attr \

