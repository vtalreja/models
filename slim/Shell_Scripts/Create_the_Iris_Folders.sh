#!/bin/bash
# A script to create individual folders for each subject from the all_in_one folder for CASIAand ND_IRIS Datasets

cd

DATA_DIR=/media/veerut/DATADISK/TensorFlow/Iris/CASIA_ND_IRIS_Train_Data/NormalizedImages

cd ${DATA_DIR}

for i in $(ls -1 | sed -e 's/\.bmp$//'|cut -d'_' -f1|cut -c 1-5 |uniq)

do

NEW_DIR=/media/veerut/DATADISK/TensorFlow/Iris/CASIA_ND_IRIS_Train_Data/"$i"

mkdir ${NEW_DIR}

cd ${DATA_DIR}

cp "$i"* $NEW_DIR
done




