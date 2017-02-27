#!/bin/bash
# A script to run the tensor flow slim with the data inputs and hyperparameters changed

cd /media/veerut/DATADISK/TensorFlow/models/slim



DATASET_DIR=/media/veerut/DATADISK/TensorFlow/Iris/CASIA_ND_IRIS_TFRecords
TRAIN_DIR=/media/veerut/DATADISK/TensorFlow/Iris/CheckPoints_Train_Logs_5_more_readers_clones_1_bn_3
EVAL_DIR=/media/veerut/DATADISK/TensorFlow/Iris/CheckPoints_Train_Logs_5_more_readers_clones_1_bn_3_eval

python eval_image_classifier.py \
   --checkpoint_path=${TRAIN_DIR} \
   --eval_dir=${EVAl_DIR} \
   --dataset_name=casia_ndiris \
   --dataset_split_name=validation \
   --dataset_dir=${DATASET_DIR} \
   --batch_size=40 \
   --New_Height_Of_Image=64 \
   --New_Width_Of_Image=512 \
   --max_num_batches=20 \
   --num_preprocessing_threads=10 \
   --model_name=vgg_16

