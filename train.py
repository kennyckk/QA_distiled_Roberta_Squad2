from preprocessing import to_data_dict,preprocess_training_examples,preprocess_validation_examples
from postprocessing import compute_metrics
from functools import partial
import torch
from transformers import AutoTokenizer,AutoModelForQuestionAnswering,default_data_collator,get_scheduler
import evaluate
from datasets import Dataset,DatasetDict
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm

def train(num_train_epochs,model,optimizer,lr_scheduler,train_dataloader,eval_dataloader,accelerator,compute_metrics,metric):

  #to get accelerator
  model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader)


  progress_bar = tqdm(range(num_train_epochs * len(train_dataloader)))

  for epoch in range(num_train_epochs):
    # Training
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        if step%20==0:
          print("This is the ep:{} steps:{}/{}, the loss for this batch is {}".format(epoch,step,len(train_dataloader),loss.item()))

    # Evaluation
    model.eval()
    start_logits = []
    end_logits = []
    accelerator.print("Evaluation!")
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(validation_dataset)]
    end_logits = end_logits[: len(validation_dataset)]

    metrics = compute_metrics(
        start_logits, end_logits, validation_dataset, raw_datasets["validation"],metric
    )
    print(f"epoch {epoch}:", metrics)

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_path, save_function=accelerator.save)

if __name__ =="__main__":
    #set the output path for saved model after training
    output_path="./saved_model"

    #load the json file and extract context,answer,question,id columns into dicts im memory
    train_dict=to_data_dict('train-v2.0.json')
    val_dict=to_data_dict('dev-v2.0.json')
    # convert the dicts into datasets object
    train_datasets=Dataset.from_dict(train_dict)
    test_datasets=Dataset.from_dict(val_dict)
    #combine the 2 datasets into one as shown in official huggingface qa course
    raw_datasets = DatasetDict({'train':train_datasets, "validation":test_datasets})

    #load in tokenizer
    model_checkpoint="distilroberta-base" #distil bert use here
    tokenizer=AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.save_pretrained(output_path+"/") #this is to save the tokenizer config to fullfil pipeline format
    #print("tokenizer saved")

    #preprocess trainign data
    train_dataset=raw_datasets['train'].map(
        partial(preprocess_training_examples,tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    #preprocess testing data
    validation_dataset = raw_datasets["validation"].map(
        partial(preprocess_validation_examples,tokenizer),
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )
    #prepare the train dataset into dataLoader
    train_dataset.set_format("torch")
    validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    validation_set.set_format("torch")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=16,
    )
    eval_dataloader = DataLoader(
        validation_set, collate_fn=default_data_collator, batch_size=16
    )

    #intiialize accelerator for distributed training
    accelerator = Accelerator(mixed_precision="fp16")

    #to configure the model as model for QA using distil-bert-architecture
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    #to config optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    #set up scheduler for learning rate optimization
    num_train_epochs = 3
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    # initialize metrices for squad2.0 data
    metric = evaluate.load("squad_v2")

    #training and show metrice for each epoches
    train(num_train_epochs,model,optimizer,lr_scheduler,train_dataloader,eval_dataloader,accelerator,compute_metrics,metric)

