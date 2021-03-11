#!/bin/bash

#### local path
SQUAD_DIR=data/mash_qa
INIT_CKPT_DIR=xlnet_cased_L-24_H-1024_A-16

#### google storage path
# GS_ROOT=.
GS_PROC_DATA_DIR=proc_data/squad_consec

python3 run_sentqa_att.py \
  --use_tpu=False \
  --do_prepro=True \
  --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
  --train_file=${SQUAD_DIR}/train_webmd_squad_v2_consec.json \
  --predict_file=${SQUAD_DIR}/val_webmd_squad_v2_consec.json \
  --output_dir=${GS_PROC_DATA_DIR} \
  --uncased=False \
  --max_seq_length=512 \
  $@

