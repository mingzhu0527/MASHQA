#!/bin/bash

#### local path
SQUAD_DIR=data/squad
INIT_CKPT_DIR=xlnet_cased_L-24_H-1024_A-16
PROC_DATA_DIR=proc_data/squad_consec
MODEL_DIR=experiment/squad_consec

#### Use 3 GPUs, each with 8 seqlen-512 samples

python3 run_sentqa_f1_57.py \
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
  --predict_batch_size=2 \
  --learning_rate=2e-5 \
  --adam_epsilon=1e-6 \
  --iterations=1000 \
  --save_steps=5000\
  --train_steps=120000 \
  --warmup_steps=1000 \
  $@