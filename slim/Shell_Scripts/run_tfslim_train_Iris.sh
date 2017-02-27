#!/bin/bash
# A script to run the tensor flow slim with the data inputs and hyperparameters changed

cd /media/veerut/DATADISK/TensorFlow/models/slim

DATASET_DIR=/media/veerut/DATADISK/TensorFlow/Iris/CASIA_ND_IRIS_TFRecords
TRAIN_DIR=/media/veerut/DATADISK/TensorFlow/Iris/CheckPoints_Train_Logs_5_more_readers_clones_1_bn_4
CHECKPOINT_PATH=/media/veerut/DATADISK/TensorFlow/Iris/CheckPoints_Train_Logs/vgg_16.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=casia_ndiris \
    --dataset_split_name=train \
    --model_name=vgg_16 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --num_clones=1 \
    --checkpoint_exclude_scopes=vgg_16/fc6/biases,vgg_16/fc6/weights,vgg_16/fc7/biases,vgg_16/fc7/weights,vgg_16/fc8/biases,vgg_16/fc8/weights \
    --ignore_missing_vars \
    --New_Height_Of_Image=64 \
    --New_Width_Of_Image=512 \
    --num_readers=2 \
    --num_preprocessing_threads=10 \
    --moving_average_decay=0.9 \
    --max_number_of_steps=150000 \
    --batch_size=40 \
    --learning_rate=0.1 \
    --save_interval_secs=1800 \
    --save_summaries_secs=1200 \
    --log_every_n_steps=100 \
    --learning_rate_decay_type=exponential \
    --optimizer=sgd \
    --weight_decay=0.00005 \
    --learning_rate_decay_factor=0.9 \
     --num_epochs_per_decay=2\
     --momentum=0.9
    


