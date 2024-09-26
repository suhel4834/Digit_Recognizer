from flask import Flask, request, render_template
import pandas as pd
from model import preprocess_image, predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    
    # Assuming the file is an image, process it
    df = preprocess_image(file)  # User-defined processing function

    new_test = df.to_numpy().reshape(-1, 28, 28, 1)

    prediction = predict(new_test)   # Prediction using the model

    return str(prediction)  # Return the prediction as a string

if __name__ == '__main__':
    app.run(debug=True)
