from flask import Flask, render_template, request
import requests

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/KnutJaegersberg/gpt2-chatbot"
HEADERS = {"Authorization": "Bearer"}


def query(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()


def func(prompt):
    output = query({"inputs": prompt})
    if output and isinstance(output, list) and len(output) > 0:
        generated_text = output[0].get('generated_text', '')
    return generated_text


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    response_text = func(user_input)
    return render_template('index.html', user_input=user_input, response_text=response_text)


if __name__ == '__main__':
    app.run(debug=True)
