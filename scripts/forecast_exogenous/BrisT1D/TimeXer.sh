#!/usr/bin/env bash
# TimeXer — BrisT1D (20 subjects, 30-min horizon)
#
# seq_len=24  : 2-hr lookback (24 × 5 min)
# label_len=6 : 30-min decoder warm-up
# pred_len=6  : 30-min forecast horizon
# patch_len=6 : 4 patches of 30 min each (patch_num = seq_len // patch_len = 4)
# enc_in=4    : [carbs, total_insulin, steps, glucose]
# c_out=1     : MS mode — predict glucose only
# --inverse   : report test RMSE in raw mg/dL
# batch_size=128 : larger dataset (411k train rows) supports bigger batches

model_name=TimeXer
dataset=BrisT1D
des='TimeXer-BGlucose'

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ${dataset}_24_6 \
  --model $model_name \
  --data $dataset \
  --root_path ./dataset/BrisT1D/ \
  --data_path bris_train.csv \
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
  --des $des \
  --inverse \
  --itr 1
