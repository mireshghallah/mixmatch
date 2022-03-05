# mixmatch
Repository for ACL 2022 paper Mix and Match: Learning-free Controllable Text Generation using Energy Language Models


# File structure

In this repo you see mix_match_code, which contains all the scripts for running generation and evaluation. Under data, you would see one compressed file, comprising of two directories, mix_match_generations and mix_match_data. The first one has the sample generations, and contains two folders, one for human evaluations against FUDGE and PPLM (human_evals_fudge_pplm), and the other for automatic evaluations, for sentiment and bias (output_samples_bias_sentiment). We have not included the data files for the formality, since the GYAFC dataset requires permission for access, so we cannot release it. 

The mix_match_data/clsf_data folder contains training samples for training the classifiers, which could be avoided if huggingface classifiers are used. mix_match_data/data contains the prompts/original sentences used for generation/transfer.

All the checkpoints of our models/classifiers are available here: https://zenodo.org/record/5855005

# Run Generation

Once you open the mix_match_code folder, place mix_match_data/data in mix_match_code/batched_MH/ and then navigate to mix_match_code/batched_MH/scripts, where you will see folders for each task. For instance, bias, is for the de-biasing task. To run the experiments, run:

```bash
sample_batched.sh
```

you can run sample_batched_boost.sh, to the the boosted version.  You can go to pplm and topic folders to get similar scripts. 

# Get Metrics

For getting all our evaluation metrics, navigate to mix_match_code/get_metrics, and you will see all evaluation scripts there. To get the metrics for bias, run:

```bash
get_abl_metrics_bias.sh
```