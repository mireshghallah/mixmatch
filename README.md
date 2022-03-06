# Mix and Match
Repository for ACL 2022 paper Mix and Match: Learning-free Controllable Text Generation using Energy Language Models

# Creating the Environment


```bash
conda create --name env --file package-list.txt
```

# File structure

In this repo you see mix_match_code, which contains all the scripts for running generation and evaluation. The sampl_generations directory contains sample generations, and has two folders, one for human evaluations against FUDGE and PPLM (human_evals_fudge_pplm), and the other for automatic evaluations, for sentiment and bias (output_samples_bias_sentiment). We have not included the data files for the formality, since the GYAFC dataset requires permission for access, so we cannot release it. 

All the  classifier checkpoints are available [here](https://zenodo.org/record/5855005).

Data for training the classifiers is available here (if you want to train your own) [here](https://drive.google.com/drive/folders/1JJE89FO4Z88fm85cmTVw1sjE7pa4Gyki?usp=sharing).

# Run Generation

Once you open the mix_match_code folder, place mix_match_data/data in mix_match_code/batched_MH/ and then navigate to mix_match_code/batched_MH/scripts, where you will see folders for each task. For instance, bias, is for the de-biasing task. To run the experiments, run:

```bash
bash bash ./mix_match_code/batched_MH/scripts/yelp/sample_batched.sh
```

Before running, *make sure you set the disc_dir to where you have placed your classifier*, or if it is an huggingface classifier, place the model name there. The outputs will be saved in a folder in the ``out_path'' you provide to the script. The opt_samples.txt is the cleaned, processed outputs. There is also metadata (energy values and other metrics) saved along in the output folder.  

For the de-biasing task, aparat from sample_batched.sh, you can run sample_batched_boost.sh, to run our generation with the agency boosting, or sample_batched_mask.sh to run the verb replacement ablation from the paper.  The PPLM folder generates with PPLM prompts using our method (for comparison with PPLM), the topic folder does topic-oriented generation (for comparison with FUDGE), the yelp folder scripts can be used for yelp sentiment transfer, and form_em can be used for formality transfer.





# Get Metrics

To get the evaluation the metrics for the de-biasing experiment, run:

```bash
bash ./mix_match_code/get_metrics/get_abl_metrics_yelp.sh
```

We have set some of our existing generations there, so when you run you will get metrics for those. You can also change it and replace it with your own generations. Run the script for other datasets/tasks to get their metrics. 


# Runing Generation for Baselines
For generating samples from the baseline, here are the commands we executed to get the baseline outputs, after we setup the environment by cloning the repositories.


For Fudge:

```bash
 python -u evaluate_topic.py --ckpt ckpt/topic/future_word_predictor/model.pth.tar --dataset_info ckpt/topic/future_word_predictor/dataset_info --prefix_file topic_data/topic_prefixes.txt --wordlist_dir topic_data/wordlists --condition_lambda 10 --verbose --precondition_topk 200 --topk 10 --sample_size 1 --max_sample_batch 1 --length_cutoff 25 --log_file topic_preds_25_lam10_1.log
```

For PPLM:

```bash
python run_pplm.py -D sentiment --length 12 --gamma 1.0 --num_iterations 10 --num_samples 20 --stepsize 0.02 --kl_scale 0.03 --gm_scale 0.90 --sample
```


# Get Human Result Evaluations

You can re-produce the human evaluation  based on the generated outputs and the turk results by running the following python scripts from the root directory of the repository:


```bash
python ./mix_match_code/batched_MH/scripts/human_eval_result_fudge.py 
python ./mix_match_code/batched_MH/scripts/human_eval_result.py 
```


# Training Classifiers

You can use huggingface classifiers, or our checkpoints provided in the link on the top of the page. However, if you want to train your own, you can download the training data from [this link](https://drive.google.com/drive/folders/1JJE89FO4Z88fm85cmTVw1sjE7pa4Gyki?usp=sharing), provide the data directory to the following script and run it.

```bash
bashe ./mix_match_code/clsf_train/run_classification_bias.sh

```

You can run other scripts in the directory to train for other tasks/datasets. 