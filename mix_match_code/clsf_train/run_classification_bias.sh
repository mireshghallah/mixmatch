python run_classification.py \
        --train_file ./data/bias/train.json \
        --validation_file ./data/bias/dev.json \
        --model_name_or_path "bert-base-uncased" \
        --do_train \
        --do_eval \
        --output_dir  ./bias \
        --max_seq_length 32 \
        --pad_to_max_length True \
        --line_by_line True \
        --use_fast_tokenizer False \
        --preprocessing_num_workers 1 \
        --learning_rate_input  1e-4 \
        --num_train_epochs 2 \
        --warmup_steps=1000 \
        --num_classes 3 \
        #--per_device_eval_batch_size 64 \
        #--per_device_train_batch_size 32 \ 
        #0.000001
        # \   --dataset_name sentiment140 \