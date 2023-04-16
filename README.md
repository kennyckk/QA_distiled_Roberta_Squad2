# QA_distiled_Roberta_Squad2

This is a QA system constructed with distil-Roberta Model and tokenizer from Huggingface. 
The model was finetuned on SQuADv2 datasets and achieved 72.3% & 75.4% exactMatch and F1 score on eval datasets.
This set of code also includes the predicted results file "distil-roberta-pred.json" on eval datasets, original training code, quick-test pipeline and also a simple UI.

this set of codes was tested in python==3.9 env
please use ```pip install -r requirements.txt ``` to install all dependencies

train.py 
Running this file will train the distilled Roberta model on SQuAD 2.0 all over again to generate saved_model folder containing the trained model paramters

Below files relied on the saved_model folder which contains trained model paramters. But the folder is too large to be uploaded here.
Please download the folder from here and put it to the same directory of the pipeline.py file:
https://drive.google.com/drive/folders/1vZbFM6B5o7XE91XY-GEYaATIO4KtHjuu?usp=sharing

pipeline.py  
Its a quick testing file for testing the functionality of the QA model
it would take input of context and a question and generate an answer in command line.

eval_predict.py
It is to execute evaluation on the dev-v2.0.json eval datasets and to output the "distil-roberta-pred.json" file which was tested on official SQuADv2 CodaLab with 72.3% & 75.4% exactMatch and F1 score

app.py
```flask run```
This file will run a flask application and create a UI at local host for using the QA system.


