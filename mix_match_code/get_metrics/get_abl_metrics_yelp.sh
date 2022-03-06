python -Wignore ./mix_match_code/get_metrics/metric.py  --checkpoint_dir sample_generations/output_samples_bias_sentiment/yelp_imdb/mix_match \
--attr_file ./data/yelp/test_li.attr \
--text_file ./data/yelp/test_li.txt \
--ref_file ./data/yelp/test_li_reference.txt \
--clsf_name /home/fmireshg/berglab.projects/sent_analysis/sent_anlys/clsf_train/yelp_cls_2/models/checkpoint-100 \
--ext_clsf \


