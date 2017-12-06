#!/bin/bash

s=4
c=0.1
e=0.3
out_name="s${s//.}c${c//.}e${e//.}"
mainDir=$(pwd)
featDir=${mainDir}"/features/"
modelDir=${mainDir}"/models/"
libLinDir="/home/tyler/liblinear-2.11"

echo "---running feature extraction---"
python3 featureExtraction18.py

echo "---converting sentences to vectors---"
python3 sent2vec18.py

cd ${libLinDir}
echo "---training LibLinear model---"
./train -s ${s} -c ${c} -e ${e} ${modelDir}"libLinearInput_train.txt" ${modelDir}${out_name}".model"
echo "---predicting on test set---"
./predict ${modelDir}"libLinearInput_test.txt" ${modelDir}${out_name}".model" ${modelDir}${out_name}"_predictions.txt"

cd ${mainDir}
echo "---adding labels to LibLinear output---"
python3 addLabels18.py ${s} ${c} ${e}
echo "---scoring model---"
perl "semeval2018_task7_scorer-v1.1.pl" ${modelDir}${out_name}"_predictions_with_labels.txt" answer_key18.txt