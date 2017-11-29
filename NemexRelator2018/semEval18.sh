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

