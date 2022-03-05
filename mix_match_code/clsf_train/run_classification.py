#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import itertools


from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process, IntervalStrategy
from transformers.utils import check_min_version

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

#from azureml.core import Run
    
import numpy as np
# Estimate the correct masked words like in Swiftkey -- not necessarily correctly
#def compute_metrics(p , mask=None):
#    print(p.predictions.shape)
#    print('Prediction Block Size: {}'.format(p.predictions.shape))
#    if len(list(p.predictions.shape))<3:
#        preds = np.argmax(p.predictions, axis=1)
#    else:
#        preds = np.argmax(p.predictions, axis=2)

#    if mask is None:
#        return {'acc': np.mean(np.float((preds == p.label_ids)))}
#    else:
#        #valid = preds >1  # reject oov predictions even if they're correct.
#        valid = mask==1
#        return {'acc': (preds.eq(p.label_ids.cpu()) * valid.cpu()).float().mean()}



def compute_metrics(pred):
    labels = pred.label_ids
    print("max labels is: ", max(labels))
    max_lab =  max(labels)
    min_lab = min(labels)
    print("min labels is ", min(labels))
    preds = pred.predictions.argmax(-1)
    if max_lab-min_lab>1:
        precision, recall, f1 = 0,0,0
    else: 
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0.dev0")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    num_classes: int = field(
        default=2,
        metadata={"help": "how many classes does the classification task have"},
    )
    learning_rate_input: float = field(
        default=0.001,
        metadata={"help": "learning rate"},
    )
    adapter: bool = field(
        default=False,
        metadata={"help": "use an adapter or not?"},
    )
    pretrained_ad: bool = field(
        default=False,
        metadata={"help": "use pretrained adapter or not?"},
    )
    fuse_n_adapters: bool = field(
        default=False,
        metadata={"help": "fuse  adapters or not?"},
    )
    model_name: str = field(
        default='super-buzzard',
        metadata={"help": "fuse  adapters or not?"},
    )
    adapter_name: str = field(
        default='adp2',
        metadata={"help": "fuse  adapters or not?"},
    )
    directory: str = field(
        default='/mnt/models/sent140/models_bertembed_partitioned_adapter',
        metadata={"help": "fuse  adapters or not?"},
    )
    partition_count: int = field(
        default=4,
        metadata={"help": "fuse  adapters or not?"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    
    ##get experiment name
    #experiment_name ='-'.join(Run.get_context().id.split('-')[2:4])
    #experiment_root = os.path.join(training_args.output_dir, experiment_name)
    model_path = os.path.join(training_args.output_dir, 'models')
    log_path   = os.path.join(training_args.output_dir, 'log')

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    # experiment name


    training_args.output_dir = model_path #os.path.join(data_path, training_args.output_dir)
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print('Found Checkpoint: ', last_checkpoint)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()


    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split= 'test' #f"train[:{data_args.validation_split_percentage}%]",
            )
            datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split='train' #f"train[{data_args.validation_split_percentage}%:]",
            )
        print ("Named dataset****",max(datasets['test']['sentiment']))
        #exit(0)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file#.split(',')
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(extension, data_files=data_files)
        print(datasets['train'])
        print(datasets['validation'])
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "num_labels":model_args.num_classes
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSequenceClassification.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    ##adapter
    if model_args.adapter:
        model.add_adapter("FTL", config={
                                    "ln_after": False,
                                    "ln_before": False,
                                    "mh_adapter": False,
                                    "output_adapter": True,
                                    "adapter_residual_before_ln": False,
                                    "non_linearity": "relu",
                                    "original_ln_after": True,
                                    "original_ln_before": True,
                                    "reduction_factor": 16,
                                    "residual_before_ln": True}
                                )
                    #Activate the adapter
        model.train_adapter("FTL")

    elif model_args.pretrained_ad:
        adapter_location = '/mnt/models/sent140/bert_cache/adapter/bert-base-uncased_sentiment_sst-2_houlsby'
        model.load_adapter(adapter_location, load_as=model_args.adapter_name,  with_head=True)

        #self.model.load_adapter("pos/ldc2012t13@vblagoje", load_as=model_args['adapter_name'],  with_head=False, cache_dir=model_args['cache_dir'])
        model.set_active_adapters(model_args.adapter_name)
        model.train_adapter(model_args.adapter_name)
        print("loading the adapter {}, pretrained".format(adapter_location))

    elif model_args.fuse_n_adapters:
            
            model_name = model_args.model_name #exciting-snake
            adapter_name= model_args.adapter_name
            model_dir = model_args.directory #models 
            partitions = model_args.partition_count

            for i in range(partitions):
                #load adapter 1:

                best_trained_model_adapter = model_dir+"/partition_{}".format(str(i))+"/"+model_name+'/models/'+adapter_name
                model.load_adapter(best_trained_model_adapter,load_as="adp{}".format(str(i)),with_head=False)
                print("done loading adapter {}".format(str(i)) )
            ######
            config ={"key": False,
                            "query":  False,
                            "value": False,
                            "query_before_ln":  False,
                            "regularization":  False,
                            "residual_before":  False,
                            "temperature": False,
                            "value_before_softmax": True,
                            "value_initialized":  False}
            #####Fuse
            adapter_setup = [["adp{}".format(str(i)) for i in range(partitions)]]
            model.add_fusion(adapter_setup[0] ) # adapter_fusion_config = config
            model.train_fusion(adapter_setup)
      
            print("done fusing adapters" )
            #print("name is *****", list(self.model.active_adapters))

            #if ('checkpoint_load' in model_args):
            #    best_trained_model_1 = model_args['checkpoint_load']+'/models/best_val_acc_model.tar'
            #    model.load_state_dict(T.load(best_trained_model_1, map_location = None if T.cuda.is_available() else T.device('cpu'))['model_state_dict'])
            #    print(f'loading checkpoint from {best_trained_model_1}, for eval', loglevel=logging.INFO )
            
        


    else:
        model.train()

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    #text_column_name = "text" if "text" in column_names else column_names[0]
    text_column_name = column_names
    print("Named columns" , column_names, text_column_name)
    #exit(0)

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warn(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warn(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False
        print("padding is", padding)
        
        def tokenize_function(examples):
            # Remove empty lines

            examples["text"] = [line[0] for line in examples["text"] if len(line[0]) > 0 and not line[0].isspace()]
       
            output =  tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                #return_special_tokens_mask=True,
            )
            

            output ['labels'] = [s[0] for s in examples['label']]  #
            #output ['labels'] = [1 if s== 4 else 0 for s in examples['sentiment']] #print("HEEEEERREEE", max(examples['sentiment']))
            
            return output

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=text_column_name,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(itertools.chain.from_iterable(examples[k])) for k in examples.keys()}
            #concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = default_data_collator #DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)
    print("dataset", eval_dataset)
    #exit(0)
    # Initialize our Trainer
    
    training_args.logging_steps=20 #
    training_args.eval_steps=1000
    training_args.save_steps=1000
    training_args.evaluation_strategy=IntervalStrategy.STEPS #
    training_args.save_strategy=IntervalStrategy.STEPS
    training_args.do_train = True
    training_args.do_predict = True
    training_args.learning_rate = model_args.learning_rate_input
    training_args.per_device_train_batch_size = 64 #32
    training_args.per_device_eval_batch_size = 64

    logger.info(f"Training/evaluation parameters {training_args}")

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    logger.info(f"Training/evaluation parameters {trainer}")
    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        print('Resuming from Checkpoint: ', last_checkpoint)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        print("Metrics: ", metrics, "**********")

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        if 0:
            # Detecting last checkpoint and loading the latest model
            checkpoint = None
            if os.path.isdir(training_args.output_dir) :
                checkpoint = get_last_checkpoint(training_args.output_dir)

            if checkpoint is not None:
                import torch

                print('Loading From Checkpoint: {}'.format(os.path.join(checkpoint, "pytorch_model.bin")))
                state_dict = torch.load(os.path.join(checkpoint, "pytorch_model.bin"))
                model.load_state_dict(state_dict)

            # Initialize our Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                compute_metrics = compute_metrics,
                train_dataset=None,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        #perplexity = math.exp(metrics["eval_loss"])
        #metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()