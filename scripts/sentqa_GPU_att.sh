#!/bin/bash

#### local path
SQUAD_DIR=data/mash_qa
INIT_CKPT_DIR=xlnet_cased_L-24_H-1024_A-16
PROC_DATA_DIR=proc_data/squad_consec
MODEL_DIR=experiment/squad_consec

python3 run_sentqa_att_sparse_hier.py \
  --use_tpu=False \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --model_config_path=${INIT_CKPT_DIR}/xlnet_config.json \
  --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
  --output_dir=${PROC_DATA_DIR} \
  --init_checkpoint=${INIT_CKPT_DIR}/xlnet_model.ckpt \
  --model_dir=${MODEL_DIR} \
  --train_file=${SQUAD_DIR}/train_webmd_squad_v2_consec.json \
  --predict_file=${SQUAD_DIR}/val_webmd_squad_v2_consec.json \
  --uncased=False \
  --max_seq_length=512 \
  --do_train=True \
  --train_batch_size=2 \
  --do_predict=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --predict_batch_size=8 \
  --learning_rate=2e-5 \
  --adam_epsilon=1e-6 \
  --iterations=10 \
  --save_steps=20\
  --train_steps=20 \
  --warmup_steps=10 \
  $@
