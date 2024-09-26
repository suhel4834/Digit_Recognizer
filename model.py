import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import io

# Load your trained model
model = load_model('digit_recognizer_model.keras')

def preprocess_image(image_path,left_cor = 60, upper_cor = 1680, right_cor = 120, lower_cor = 1775):
  img = Image.open(image_path)
  crop_box = (left_cor, upper_cor, right_cor, lower_cor)
  cropped_img = img.crop(crop_box)

  cropped_img.save('cropped_image.jpg')
  image_path = 'cropped_image.jpg'
  img = cv2.imread(image_path)

  # Convert to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  #Apply thresholding to make the digits stand out
  _, thresh = cv2.threshold(gray, 120, 250, cv2.THRESH_BINARY_INV)
  
  # Get the height and width of the image
  height, width = thresh.shape

  # Calculate the height of each cropped part
  crop_height = height // 3

  # Crop the image into 3 equal parts vertically
  cropped_images = []
  for i in range(3):
    cropped_image = thresh[i * crop_height : (i + 1) * crop_height, :]
    cropped_images.append(cropped_image)

  # Resize each cropped image to 28x28 and store in a DataFrame
  image_data = []
  for cropped_img in cropped_images:
    resized_img = cv2.resize(cropped_img, (28, 28))
    # Normalize the pixel values by dividing by 255.0
    normalized_img = resized_img / 255.0

    # Flatten the 28x28 image into a 1D array (784 elements)
    flattened_img = normalized_img.flatten()

    # Append the flattened image to the image_data list
    image_data.append(flattened_img)
  
  # Convert the list of image data to a DataFrame (each row is a digit image)
  df = pd.DataFrame(image_data)

  return df

def predict(df):
    # Make predictions
    prediction = model.predict(df)
    return np.argmax(prediction, axis=1)  # Return the predicted class
