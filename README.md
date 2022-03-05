# Mix and Match
Repository for ACL 2022 paper Mix and Match: Learning-free Controllable Text Generation using Energy Language Models

# Creating the Environment


```bash
conda create --name env --file package-list.txt
```

# File structure

In this repo you see mix_match_code, which contains all the scripts for running generation and evaluation. The sampl_generations directory contains sample generations, and has two folders, one for human evaluations against FUDGE and PPLM (human_evals_fudge_pplm), and the other for automatic evaluations, for sentiment and bias (output_samples_bias_sentiment). We have not included the data files for the formality, since the GYAFC dataset requires permission for access, so we cannot release it. 

The mix_match_data/clsf_data folder contains training samples for training the classifiers, which could be avoided if huggingface classifiers or our own  classifiers (checkpoints available here: https://zenodo.org/record/5855005) are used.

# Run Generation

Once you open the mix_match_code folder, place mix_match_data/data in mix_match_code/batched_MH/ and then navigate to mix_match_code/batched_MH/scripts, where you will see folders for each task. For instance, bias, is for the de-biasing task. To run the experiments, run:

```bash
sample_batched.sh
```

you can run sample_batched_boost.sh, to the the boosted version.  The PPLM folder generates with PPLM prompts using our method (for comparison with PPLM), the topic folder does topic-oriented generation (for comparison with FUDGE), the yelp folder scripts can be used for yelp sentiment transfer, and form_em can be used for formality transfer.







# Get Metrics

For getting all our evaluation metrics, navigate to mix_match_code/get_metrics, and you will see all evaluation scripts there. To get the metrics for bias, run:

```bash
get_abl_metrics_bias.sh
```