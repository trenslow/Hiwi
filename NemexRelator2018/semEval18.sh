#!/bin/bash

if [ $# == 1 ]; then  # doesnt work yet
    echo "Doing own $1-fold cross-validation"
    crossVal=true
    k=$1
elif [ $# -gt 1 ]; then
    echo "exiting: too many arguments"
    exit 1
else
    echo "doing training/test on subset given by SemEval"
    crossVal=false
    k=1
fi

s=4
c=0.1
e=0.3
out_name="s${s//.}c${c//.}e${e//.}"
mainDir=$(pwd)
featDir=${mainDir}"/features/"
modelDir=${mainDir}"/models/"
libLinDir="/home/tyler/liblinear-2.11"

resultsDir=${mainDir}"/results/"
echo "---clearing results directory---"
rm results/results*.txt

for i in $(seq 1 ${k});
do
    echo "---running feature extraction---"
    python3 featureExtraction18.py ${crossVal}
    ret=$?
    if [ ${ret} == 1 ]; then
         exit 1
    fi

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

    if [ ${crossVal} = true ]; then
        echo "---writing results to file---"
        perl "semeval2018_task7_scorer-v1.1.pl" ${modelDir}${out_name}"_predictions_with_labels.txt" answer_key18.txt > ${resultsDir}"results${i}.txt"
    else
        echo "---scoring model---"
        perl "semeval2018_task7_scorer-v1.1.pl" ${modelDir}${out_name}"_predictions_with_labels.txt" answer_key18.txt
    fi
done

if [ ${crossVal} = true ]; then
    python3 average.py
fi