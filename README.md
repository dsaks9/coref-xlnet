# XLNet for Coreference Resolution
This repository contains code for 'XLNet for Coreference Resolution'. 
The model architecture itself is an extension of the SpanBert based coref model.

## Setup
* Install python3 requirements: `pip install -r requirements.txt`
* `export data_dir=</path/to/data_dir>`
* `./setup_all.sh`: This builds the custom kernels
* Clone xlnet repo from https://github.com/zihangdai/xlnet.git


## Setup Instructions

This assumes access to OntoNotes 5.0.
`./setup_training.sh <ontonotes/path/ontonotes-release-5.0> $data_dir`. This preprocesses the OntoNotes corpus, and downloads the original XLNet models. The models which will be finetuned using `train_xlnet.py`. 

* Experiment configurations are found in `experiments.conf`. Choose an experiment that you would like to run.
* Finetuning: `GPU=0 python train_xlnet.py <experiment>`
* Results are stored in the `log_root` directory (see `experiments.conf`) and can be viewed via TensorBoard.
* Evaluation: `GPU=0 python evaluate_xlnet.py <experiment>`. This currently evaluates on the dev set.

## Finetuning

The following shows an example of using train_xlnet.py to finetune XLNet for coreference resolution task.
```
os.environ['data_dir'] = "../xlnet_data_dir/"

! GPU=0 python train_xlnet.py train_xlnet_768 \
  --do_train=True \
  --do_eval=False \
  --uncased=False \
  --spiece_model_file=$data_dir/xlnet_cased_L-12_H-768_A-12/spiece.model \
  --model_config_path=$data_dir/xlnet_cased_L-12_H-768_A-12/xlnet_config.json \
  --init_checkpoint=$data_dir/xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt \
  --max_seq_length=768 \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --learning_rate=5e-05 \
  --train_steps=80000 \
  --warmup_steps=500 \
  --dropout=0.2 
```
