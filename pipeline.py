from transformers import pipeline

# load in the saved model and parameters into hugging face pipeline
model_checkpoint="./saved_model"
question_answerer= pipeline("question-answering", model=model_checkpoint)

if __name__=="__main__":
    while True:
        context = str(input("please input your context!:"))
        question= str(input("please input your question:"))

        print(question_answerer(question=question, context=context, handle_impossible_answer=True))

    #some sample question and context to use
    # question = "Which deep learning libraries back 🤗 Transformers?"
    # context = """
    # 🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
    # between them. It's straightforward to train your models with one before loading them for inference with the other.
    # """
