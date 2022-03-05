python  -Wignore ../batched_MH/scripts/jx_cls/cnn_classify.py  --run_classification_eval --file_dir $1 --classifier_dir $2  \
--train_src_file /home/user/dir.projects/sent_analysis/deep-latent-sequence-model/data/form_em/test.txt \
--train_trg_file /home/user/dir.projects/sent_analysis/deep-latent-sequence-model/data/form_em/test.attr \
--dev_src_file  /home/user/dir.projects/sent_analysis/deep-latent-sequence-model/data/form_em/test.txt \
--dev_trg_file  /home/user/dir.projects/sent_analysis/deep-latent-sequence-model/data/form_em/test.attr \
--dev_trg_ref /home/user/dir.projects/sent_analysis/deep-latent-sequence-model/data/form_em/test.txt \
--src_vocab  /home/user/dir.projects/sent_analysis/deep-latent-sequence-model/data/form_em/text.vocab \
--trg_vocab /home/user/dir.projects/sent_analysis/deep-latent-sequence-model/data/form_em/attr.vocab \