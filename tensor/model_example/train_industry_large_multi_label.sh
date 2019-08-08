#!/bin/bash

# python train_industry_large_multi_label.py \
# --train_data_path="data/train_industry_multi_label_large.tfrecord" \
# --test_data_path="data/test_industry_multi_label_large.tfrecord" \
# --pretrained='best_cbert_6l_8h/best_model' \
# --model_log_dir='logs_industry_multi_label_large' \
# --tag_size=241 \
# --learning_rate=5e-5

python train_industry_large_multi_label.py \
--train_data_path="data/train_multi_labels.tfrecord" \
--test_data_path="data/test_multi_labels.tfrecord" \
--pretrained='best_cbert_6l_8h/best_model' \
--model_log_dir='logs_industry_multi_label_large' \
--tag_size=238 \
--learning_rate=5e-5