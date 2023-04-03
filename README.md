# QA_distiled_Roberta_Squad2

this set of codes was tested in python==3.9 env
please use ```pip install -r requirements.txt ``` to install all dependencies

train.py 
Running this file will train the distilled Roberta model on SQuAD 2.0 all over again to generate saved_model folder containing the trained model paramters

pipeline.py  
Its a simple testing file for testing the functionality/accuracy of the model
it would take input of context and a question and generate an answer
it relied on the saved_model folder which contains trained model paramters. But the folder is too large to be uploaded here.
Please download the folder from here and put it to the same directory of the pipeline.py file:
https://drive.google.com/drive/folders/1vZbFM6B5o7XE91XY-GEYaATIO4KtHjuu?usp=sharing


