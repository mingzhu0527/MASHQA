#!/bin/bash

#### local path
SQUAD_DIR=data/webmdqa_new
INIT_CKPT_DIR=xlnet_cased_L-24_H-1024_A-16

#### google storage path
# GS_ROOT=.
GS_PROC_DATA_DIR=proc_data/squad_nonconsec_small #squad_consec squad_full squad_full_small squad_nonconsec squad_nonconsec_small

python3 run_sentqa_att.py \
  --use_tpu=False \
  --do_prepro=True \
  --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
  --train_file=${SQUAD_DIR}/train_webmd_squad_v2_nonconsec.json \
  --predict_file=${SQUAD_DIR}/small_val_webmd_squad_v2_nonconsec.json \
  --output_dir=${GS_PROC_DATA_DIR} \
  --uncased=False \
  --max_seq_length=512 \
  $@


# python3 run_sentqa_att.py \
#   --use_tpu=False \
#   --do_prepro=True \
#   --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
#   --train_file=${SQUAD_DIR}/val_webmd_squad_v2_consec.json \
#   --output_dir=${GS_PROC_DATA_DIR} \
#   --uncased=False \
#   --max_seq_length=512 \
#   $@

#### Potential multi-processing version
# NUM_PROC=8
# for i in `seq 0 $((NUM_PROC - 1))`; do
#   python run_squad.py \
#     --use_tpu=False \
#     --do_prepro=True \
#     --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
#     --train_file=${SQUAD_DIR}/train-v2.0.json \
#     --output_dir=${GS_PROC_DATA_DIR} \
#     --uncased=False \
#     --max_seq_length=512 \
#     --num_proc=${NUM_PROC} \
#     --proc_id=${i} \
#     $@ &
# done
