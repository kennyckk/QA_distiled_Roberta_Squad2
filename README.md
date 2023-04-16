# QA_distiled_Roberta_Squad2

This is a QA system constructed with distil-Roberta Model and tokenizer from Huggingface. 
The model was finetuned on SQuADv2 datasets and achieved 72.3% & 75.4% exactMatch and F1 score on eval datasets.
This set of code also includes the predicted results file "distil-roberta-pred.json" on eval datasets, original training code, quick-test pipeline and also a simple UI.

this set of codes was tested in python==3.9 env
please use ```pip install -r requirements.txt ``` to install all dependencies

<h3>distil-roberta-pred.json</h3>
This file is the predicted result on the eval dataset and was tested on official SQuADv2 CodaLab with 72.3% & 75.4% exactMatch and F1 score

<h3>train.py</h3> 
Running this file will train the distilled Roberta model on SQuAD 2.0 all over again to generate saved_model folder containing the trained model paramters

-------------------------------------------------------------------------------------------------
Below files relied on the saved_model folder which contains trained model paramters. But the folder is too large to be uploaded here.
Please download the folder from here and put it to the same directory of the following files:
https://drive.google.com/drive/folders/1vZbFM6B5o7XE91XY-GEYaATIO4KtHjuu?usp=sharing

<h3>pipeline.py</h3>  
Its a quick testing file for testing the functionality of the QA model
it would take input of context and a question and generate an answer in command line.

<h3>eval_predict.py</h3> 
It is to execute evaluation on the dev-v2.0.json eval datasets and to output the "distil-roberta-pred.json"

<h3>app.py</h3>
Use ```flask run``` to run a flask application and create a UI at local host for using the QA system.


