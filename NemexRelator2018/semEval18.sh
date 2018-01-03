#!/bin/bash

s=4
c=0.1
e=0.3
out_name="s${s//.}c${c//.}e${e//.}"
mainDir=$(pwd)
featDir=${mainDir}"/features/"
modelDir=${mainDir}"/models/"
libLinDir="/home/tyler/liblinear-2.11"
resultsDir=${mainDir}"/results/"

if [ $1 = 0 ]; then
    echo "doing training/test on subset given by SemEval organizers"
    echo "---running feature extraction---"
    python3 featureExtraction18.py 0

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
else
    echo "Doing 10-fold cross-validation"
    echo "---clearing results directory---"
    rm results/results*.txt
    for i in $(seq 1 10);
    do
    echo "---running feature extraction---"
    python3 featureExtraction18.py ${i}
    
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

    echo "---writing results to file---"
    perl "semeval2018_task7_scorer-v1.1.pl" ${modelDir}${out_name}"_predictions_with_labels.txt" answer_key18.txt > ${resultsDir}"results${i}.txt"
    done
    python3 average.py

fi
