from preprocessing import to_data_dict,preprocess_validation_examples
from postprocessing import prediction
from functools import partial
import torch
from transformers import AutoTokenizer,AutoModelForQuestionAnswering,default_data_collator,get_scheduler
from datasets import Dataset
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import json

def eval_pred(model,eval_dataloader,validation_dataset,prediction):

    # no need to set epoch and only do once
    # Evaluation
    model.eval()
    start_logits = []
    end_logits = []

    #extract the data batch by batch and feed into model
    for batch in tqdm(eval_dataloader):
        input_ids=batch["input_ids"].to(device)
        attention_mask= batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        start_logits.append(outputs.start_logits.cpu().numpy())
        end_logits.append(outputs.end_logits.cpu().numpy())

    # concatenate all resultant start/end logits from every batch to 2 single np arrays
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(validation_dataset)]
    end_logits = end_logits[: len(validation_dataset)]

    # run the predictio function from postprocessing to get a dict object of predicted results
    results = prediction(start_logits, end_logits, validation_dataset, test_datasets)

    #print("the predicted example # is: ",len(results.keys()))

    return results

if __name__ =="__main__":

    #put device on CUDA if exists
    device= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #load the json file and extract context,answer,question,id columns into dicts im memory
    val_dict=to_data_dict('dev-v2.0.json')
    # convert the dicts into datasets object
    test_datasets=Dataset.from_dict(val_dict)

    #just for small size testing
    #test_datasets=test_datasets.select(range(50))

    #load in tokenizer
    model_checkpoint="./saved_model" #used the trained model & tokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_checkpoint)

    #preprocess testing data
    validation_dataset = test_datasets.map(
        partial(preprocess_validation_examples,tokenizer),
        batched=True,
        remove_columns=test_datasets.column_names,
    )
    #prepare the train dataset into dataLoader
    validation_to_loader = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    validation_to_loader.set_format("torch")

    eval_dataloader = DataLoader(
        validation_to_loader, collate_fn=default_data_collator, batch_size=16
    )

    #to configure the model using trained parameters and move to GPU if exists
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint).to(device)

    #feed in data into model and get the resultant dictionary object for the predicted results
    results=eval_pred(model,eval_dataloader,validation_dataset,prediction)

    # to convert the dict obj to json obj and save it
    json_results = json.dumps(results)
    with open("./distil-roberta-pred.json","w") as f:
        f.write(json_results)