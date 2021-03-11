#!/bin/bash

#### local path
SQUAD_DIR=data/squad
INIT_CKPT_DIR=xlnet_cased_L-24_H-1024_A-16
PROC_DATA_DIR=proc_data/squad_consec #squad_consec_small squad_consec squad_full_newest
MODEL_DIR=experiment/squad_full_att_entmax_hier_weight_10_self_att
# MODEL_DIR=experiment/squad_consec_att python3 run_sentqa_att_softatt_gelu.py
# MODEL_DIR=experiment/squad_consec_f1_65 python3 run_sentqa_f1_65.py
# PROC_DATA_DIR=proc_data/squad_consec_small MODEL_DIR=experiment/squad_consec_att_simple python3 run_sentqa_att.py
# PROC_DATA_DIR=proc_data/squad_consec MODEL_DIR=experiment/squad_consec_att_simple python3 run_sentqa_att.py
# MODEL_DIR=experiment/squad_consec_att_simple_with_softmax python3 run_sentqa_att_with_softmax.py
# MODEL_DIR=experiment/squad_consec_att_simple_with_self python3 run_sentqa_att_with_self.py
# MODEL_DIR=experiment/squad_consec_f1_65_with_comments python3 run_sentqa_f1_65_with_comments.py
# MODEL_DIR=experiment/squad_consec python3 run_sentqa_f1_65.py BEST PERFROMANCE 65 F1
# MODEL_DIR=experiment/squad_consec_att_simple_with_softmax_parameterized python3 run_sentqa_att_with_softmax_scale_4096_parameterized.py
# MODEL_DIR=experiment/squad_consec_att_simple_with_softmax_parameterized_weight_decay python3 run_sentqa_att_with_softmax_scale_4096_parameterized_weight_decay.py
# MODEL_DIR=experiment/squad_consec_f1_65_sparse_gumbel_webmd python3 run_sentqa_f1_65_sparse_gumbel.py
# MODEL_DIR=experiment/squad_consec_att_entmax_hier python3 run_sentqa_att_sparse_hier.py
# MODEL_DIR=experiment/squad_consec_att_entmax_hier_weight_5 python3 run_sentqa_att_sparse_hier_weight_5.py
# MODEL_DIR=experiment/squad_full_att_entmax_hier_weight_10 python3 run_sentqa_att_sparse_hier_weight_10.py

#### Use 3 GPUs, each with 8 seqlen-512 samples

python3 run_sentqa_att_sparse_hier_weight_10_self_att.py \
  --use_tpu=False \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --model_config_path=${INIT_CKPT_DIR}/xlnet_config.json \
  --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
  --output_dir=${PROC_DATA_DIR} \
  --init_checkpoint=${INIT_CKPT_DIR}/xlnet_model.ckpt \
  --model_dir=${MODEL_DIR} \
  --train_file=${SQUAD_DIR}/train_webmd_squad_v2_consec.json \
  --predict_file=${SQUAD_DIR}/small_val_webmd_squad_v2_consec.json \
  --predict_dir=results \
  --uncased=False \
  --max_seq_length=512 \
  --do_train=True \
  --train_batch_size=2 \
  --do_predict=True \
  --do_eval=True \
  --eval_all_ckpt=False \
  --predict_batch_size=8 \
  --learning_rate=2e-5 \
  --adam_epsilon=1e-6 \
  --iterations=1000 \
  --save_steps=5000\
  --train_steps=200000 \
  --warmup_steps=1000 \
  $@
