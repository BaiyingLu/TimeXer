#!/usr/bin/env bash
# Test the combined-trained model on BrisT1D test split.

model_name=TimeXer
model_id=combined_24_6
des='TimeXer-BGlucose-combined'

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --model_id ${model_id} \
  --model ${model_name} \
  --data combined \
  --root_path ./dataset/BrisT1D/ \
  --data_path bris_train.csv \
  --scaler_root_path ./dataset/combined/ \
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
  --des ${des} \
  --inverse \
  --itr 1
