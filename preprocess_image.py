import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def preprocess_image(image_path,left_cor, upper_cor, right_cor, lower_cor):
  img = Image.open(image_path)
  crop_box = (left_cor, upper_cor, right_cor, lower_cor)
  cropped_img = img.crop(crop_box)
  # Display the cropped image
  # plt.imshow(cropped_img)
  # plt.axis('off')  # Hide axis
  # plt.show()
  # Save the cropped image (optional)
  cropped_img.save('cropped_image.jpg')
  image_path = 'cropped_image.jpg'
  img = cv2.imread(image_path)
  # Convert to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  #Apply thresholding to make the digits stand out
  _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
  
  # Show the preprocessed image
  #plt.imshow(thresh, cmap='gray')
  #plt.title('Preprocessed Image')
  #plt.show()

  # Use morphological operations to detect horizontal lines
  kernel = np.ones((1, 50), np.uint8)  # Horizontal kernel for detecting long lines
  horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
  
  # Find contours in the horizontal line image
  contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
  # Sort contours based on their y-coordinate (top to bottom)
  contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[1])

  # List to store the cropped images
  cropped_digits = []

  # Extract regions between the horizontal lines (3 segments)
  for i in range(len(contours) - 1):
    
    # Get bounding rectangles for the contours
    x, y, w, h = cv2.boundingRect(contours[i])
    x_next, y_next, w_next, h_next = cv2.boundingRect(contours[i + 1])
    
    # Crop the image between the horizontal lines
    digit_img = thresh[y:y_next, x:x+w]  
    
    # Crop based on horizontal line positions
    cropped_digits.append(digit_img)

    # Optional: Display each cropped digit image
    plt.imshow(digit_img, cmap='gray')
    plt.title(f'Cropped Digit {i + 1}')
    plt.show()



image_path = '181002.jpg'
left = 60
upper = 1680
right = 120
lower = 1775
df = preprocess_image(image_path, left, upper, right, lower)