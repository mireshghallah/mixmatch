python run_classification.py \
        --train_file ./data/form_em/train.json \
        --validation_file ./data/form_em/dev.json \
        --model_name_or_path "bert-base-uncased" \
        --do_train \
        --do_eval \
        --output_dir  ./form_em_2 \
        --max_seq_length 32 \
        --pad_to_max_length True \
        --line_by_line True \
        --use_fast_tokenizer False \
        --preprocessing_num_workers 1 \
        --learning_rate_input  1e-5 \
        --num_train_epochs 3 \
        --warmup_steps=1000 \
        --num_classes 2 \
        #--per_device_eval_batch_size 64 \
        #--per_device_train_batch_size 32 \ 
        #0.000001
        # \   --dataset_name sentiment140 \