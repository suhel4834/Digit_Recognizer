from flask import Flask, request, render_template
import pandas as pd
from model import preprocess_image, predict
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']

    # Save the uploaded file to a temporary location
    temp_image_path = os.path.join('temp', file.filename)
    os.makedirs('temp', exist_ok=True)  # Ensure the temp directory exists
    file.save(temp_image_path)
    
    # Assuming the file is an image, process it
    df = preprocess_image(temp_image_path)  # User-defined processing function

    os.remove(temp_image_path)

    new_test = df.to_numpy().reshape(-1, 28, 28, 1)

    prediction = predict(new_test)   # Prediction using the model

    return str(prediction)  # Return the prediction as a string

if __name__ == '__main__':
    app.run(debug=True)
