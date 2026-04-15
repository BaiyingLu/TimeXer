#!/usr/bin/env bash
# TimeXer — Population model trained on OhioT1DM + BrisT1D + HUPA-UCM combined.
#
# Train and validate on the merged splits in dataset/combined/.
# Scalers (per-participant exo + global glucose) are saved to
# dataset/combined/scalers.pkl and reused at test time.
#
# After training, run the three test scripts in this directory to evaluate
# on each dataset separately using --is_training 0.

model_name=TimeXer
model_id=combined_24_6
des='TimeXer-BGlucose-combined'

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ${model_id} \
  --model ${model_name} \
  --data combined \
  --root_path ./dataset/combined/ \
  --data_path combined_train.csv \
  --features MS \
  --target glucose \
  --freq t \
  --seq_len 24 \
  --label_len 6 \
  --pred_len 6 \
  --patch_len 6 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --e_layers 2 \
  --d_layers 1 \
  --n_heads 4 \
  --d_model 128 \
  --d_ff 256 \
  --factor 3 \
  --dropout 0.1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 20 \
  --patience 5 \
  --des ${des} \
  --inverse \
  --itr 1
