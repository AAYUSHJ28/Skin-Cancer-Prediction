# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:41:13 2024

@author: HP
"""


import cv2
import numpy as np
from keras.models import load_model

img_size = (180, 180) 

CLASSES = ["actinic keratosis","basal cell carcinoma", 
           "dermatofibroma", "melanoma", "nevus", 
           "pigmented benign keratosis", "seborrheic keratosis", 
           "squamous cell carcinoma", "vascular lesion"]

# Load the model
model1 = load_model('my_keras_model.keras')

# Read the image

image_path = "C:/Users/HP/OneDrive/Desktop/TCS/Skin Cancer Prediction/Skin cancer ISIC The International Skin Imaging Collaboration/Test/nevus/ISIC_0000003.jpg"

#image_path = "C:/Users/HP/OneDrive/Desktop/TCS/Skin Cancer Prediction/Skin cancer ISIC The International Skin Imaging Collaboration/Test/pigmented benign keratosis/ISIC_0024324.jpg"


image = cv2.imread(image_path)

if image is None:
    raise ValueError("Image not found or unable to load.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_resized = cv2.resize(image_rgb, img_size)

img = np.expand_dims(image_resized, axis=0)

# Make prediction
pred = model1.predict(img)

# Get the predicted class
predicted_class_index = np.argmax(pred)
predicted_class = CLASSES[predicted_class_index]

image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


cv2.putText(image_bgr, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Show the image with the prediction
cv2.imshow('Skin Disease Detection', image_bgr)
print(f"Predicted class: {predicted_class}")

cv2.waitKey(0)
cv2.destroyAllWindows()