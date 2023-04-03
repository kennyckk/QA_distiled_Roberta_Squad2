import json


def to_data_dict(path):
  with open(path, 'rb') as f:
    squad=json.load(f)

  raw_dict={'context':[],
        'question':[],
        'answers':[],
        'id':[]}
  for group in squad['data']:
    for passage in group['paragraphs']:
      context=passage['context']
      for qa in passage['qas']:
        question=qa['question']
        id=qa['id']
        raw_dict['context'].append(context)
        raw_dict['question'].append(question)
        raw_dict['id'].append(id)
        text_holder=[]
        ans_holder=[]
        for answer in qa['answers']:
          text_holder.append(answer['text'])
          ans_holder.append(answer['answer_start'])
        ans_dict={'text':text_holder, 'answer_start':ans_holder}
        raw_dict['answers'].append(ans_dict)
  return raw_dict


# preprocessing for the datasets to get start and end token positions for each span of inputs
def preprocess_training_examples(tokenizer,examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # assign CLS token position to start and end if ans not exist
        if len(answer['answer_start'])==0:
          start_positions.append(0)
          end_positions.append(0)
        else:
          start_char = answer["answer_start"][0]
          end_char = answer["answer_start"][0] + len(answer["text"][0])
        # If the answer is not fully inside the context, label is (0, 0)
          if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
              start_positions.append(0)
              end_positions.append(0)
          else:
              # Otherwise it's the start and end token positions
              idx = context_start
              while idx <= context_end and offset[idx][0] <= start_char:
                  idx += 1
              start_positions.append(idx - 1)

              idx = context_end
              while idx >= context_start and offset[idx][1] >= end_char:
                  idx -= 1
              end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


# pre process validation datasets making use of id and making the offsets mapping None for non context in each span
def preprocess_validation_examples(tokenizer,examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs