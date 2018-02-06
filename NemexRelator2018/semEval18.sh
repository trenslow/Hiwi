#!/bin/bash

s=1
c=0.1
e=0.1
out_name="s${s//.}c${c//.}e${e//.}"
mainDir=$(pwd)
featDir=${mainDir}"/features/"
modelDir=${mainDir}"/models/"
libLinDir="/home/tyler/liblinear-2.11"
resultsDir=${mainDir}"/results/"

if [ "$#" -ne 1 ]; then
    echo "script requires one argument, k, for development.
    k>0 does k-fold cross val, k=0 uses dev set, k=-1 for submission"
    echo "exiting..."
    exit 1
else
    k=$1
    echo "---clearing record files---"
    rm features/record*.txt
fi

if [[ "$k" = "0" ]]; then
    echo "doing training/test on subset given by SemEval organizers"
    echo "---running feature extraction---"
    python3 featureExtraction18.py ${k}

    echo "---converting sentences to vectors---"
    python3 sent2vec18.py ${k}

    cd ${libLinDir}
    echo "---training LibLinear model---"
    ./train -s ${s} -c ${c} -e ${e} ${modelDir}"libLinearInput_train.txt" ${modelDir}${out_name}".model"
    echo "---predicting on test set---"
    ./predict ${modelDir}"libLinearInput_test.txt" ${modelDir}${out_name}".model" ${modelDir}"predictions.txt"

    cd ${mainDir}
    echo "---adding labels to LibLinear output---"
    python3 addLabels18.py ${k}

    echo "---scoring model---"
    perl "semeval2018_task7_scorer-v1.2.pl" ${modelDir}"predictions_with_labels.txt" answer_key18.txt

elif [[ "$k" = "-1"  ]]; then
    echo "Creating file for competition submission"
    echo "---running feature extraction---"
    python3 featureExtraction18.py ${k}

    echo "---converting sentences to vectors---"
    python3 sent2vec18.py "0"

    cd ${libLinDir}
    echo "---training LibLinear model---"
    ./train -s ${s} -c ${c} -e ${e} ${modelDir}"libLinearInput_train.txt" ${modelDir}${out_name}".model"
    echo "---predicting on test set---"
    ./predict ${modelDir}"libLinearInput_test.txt" ${modelDir}${out_name}".model" ${modelDir}"predictions.txt"

    cd ${mainDir}
    echo "---adding labels to LibLinear output---"
    python3 addLabels18.py ${k}

    echo "Don't forget to add the task number to the first line of prediction file before submission!!"

elif [[ "$k" >  "0" ]]; then
    echo "Doing ${k}-fold cross-validation"
    echo "---clearing results directory---"
    rm results/results*.txt

    for i in $(seq 1 ${1});
    do
    echo "---current fold: ${i}---"
    echo "---running feature extraction---"
    python3 featureExtraction18.py ${k}

    echo "---converting sentences to vectors---"
    python3 sent2vec18.py ${i}

    cd ${libLinDir}
    echo "---training LibLinear model---"
    ./train -s ${s} -c ${c} -e ${e} ${modelDir}"libLinearInput_train.txt" ${modelDir}${out_name}".model"
    echo "---predicting on test set---"
    ./predict ${modelDir}"libLinearInput_test.txt" ${modelDir}${out_name}".model" ${modelDir}"predictions.txt"

    cd ${mainDir}
    echo "---adding labels to LibLinear output---"
    python3 addLabels18.py ${i}

    echo "---writing results to file---"
    perl "semeval2018_task7_scorer-v1.2.pl" ${modelDir}"predictions_with_labels.txt" answer_key18.txt > ${resultsDir}"results${i}.txt"
    done

    python3 average.py
fi
