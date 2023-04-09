"""app.py
This file is the Flask backend logic containing all necessary functions, methods and BertClassifier model
to operate the UI system.
"""
from flask import Flask, request, render_template
from transformers import BertTokenizer,BertForSequenceClassification
import torch
import numpy as np


# Declare a Flask app
app = Flask(__name__)
# to detect the device if it has CUDA or not
device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
# load in Json object for Id mapping to Class text.
class_map = {0:'anger', 1:'fear', 2:'joy', 3:'love',4: 'sadness', 5:'surprise'}
emoji_map= {
      "fear":"😖",
      "love":"😍",
      "anger":"😡",
      "joy":"😆",
      "surprise":"😲",
      "sadness":"😭"
}
#load in model and tokenizer here as global variable

model= BertForSequenceClassification.from_pretrained("./parameters/BertClassifier_ep2/")
model.to(device)
model.eval()

#load in tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#preprocess and tokenizing function
def preprocess(text):
    encoded_dict = tokenizer.encode_plus(text,
                                         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                         max_length=tokenizer.model_max_length,  # max_length set to model default
                                         truncation=True,  # Pad & truncate all sentences.
                                         pad_to_max_length=True,
                                         return_attention_mask=True,  # Construct attn. masks.
                                         return_tensors='pt',  # return pytorch tensor (len(texts), 512)
                                         )
    input_ids=(encoded_dict['input_ids'])  # returned tensor is (1,512) matching input format of model
    attention_mask=(encoded_dict['attention_mask'])

    return input_ids.to(device), attention_mask.to(device)

def load_in_model(model,input_ids, attention_mask, class_map):
    with torch.no_grad(): #make sure no grad is calculated during forward pass speed up the model
        outputs = model(input_ids, attention_mask)
        logits=outputs.logits #extract the logits of the class probability
    logits= logits.detach().cpu().numpy() #create a new tensor and move it to cpu and convert to np array
    pred_id=np.argmax(logits, axis=1).item() #find the index with largest probability

    return class_map[pred_id],emoji_map[class_map[pred_id]] #map the predicted index to the label map containing the class text




# Main function here
@app.route('/', methods=['GET', 'POST'])
def main():
    prediction=""
    pred_emoji=""
    text_input=""
    # If a form is submitted
    if request.method == "POST":
        #to get the values from the input bars
        text_input = request.form.get("text_inputs")
        #preprocess the input to proper form of inputs to BERT model
        input_ids, attention_mask=preprocess(text_input)

        #load the data to model and predict
        prediction,pred_emoji=load_in_model(model,input_ids,attention_mask,class_map)



    return render_template("index.html", output=prediction,emojii=pred_emoji,last_input=text_input)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)