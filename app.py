"""app.py
This file is the Flask backend logic containing all necessary functions, methods and Distilled Roberta model
to operate the UI system.
"""
from flask import Flask, request, render_template
from transformers import pipeline

# Declare a Flask app
app = Flask(__name__)
# load in the saved model and parameters into hugging face pipeline
model_checkpoint="./saved_model"
question_answerer= pipeline("question-answering", model=model_checkpoint)

# Main function here
@app.route('/', methods=['GET', 'POST'])
def main():
    context=""
    question=""
    answer=""
    score=0
    # If a form is submitted
    if request.method == "POST":
        #to get the values from the input bars
        context = request.form.get("context_inputs")
        question = request.form.get("question_inputs")

        context = str(input("please input your context!:"))
        question = str(input("please input your question:"))

        result = question_answerer(question=question, context=context, handle_impossible_answer=True)
        answer = result['answer']
        score = result['score']
    return render_template("index.html", output=answer, score=score)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)