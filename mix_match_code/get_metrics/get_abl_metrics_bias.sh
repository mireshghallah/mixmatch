
python -Wignore  ./mix_match_code/get_metrics/metric.py  --checkpoint_dir ./sample_generations/output_samples_bias_sentiment/bias/mix_match_best_42_54 \
--attr_file ./data/bias/test_bi.attr \
--text_file ./data/bias/test_bi.txt \
--ref_file  ./data/bias/test_bi.txt \
--clsf_name /home/fmireshg/berglab.projects/sent_analysis/sent_anlys/clsf_train/bias/models/checkpoint-1050 \

