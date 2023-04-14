"""
Bottle Classifier API

This API has one endpoint, /bottle_classify, which takes an image file as input and returns a prediction of whether 
the bottle is accepted or rejected.

Also save the images uploaded to the API to the disk for future use.
"""

from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import random

app = FastAPI()

# Constants
model = tf.keras.models.load_model('model_07_0.87.h5')
labels = ['Accepted', 'Rejected']

@app.post("/bottle_classify")
async def bottle_classify(file: UploadFile = File(...)):
    # Read the image
    image = await file.read()
    
    # Save the image to disk
    file_path = f"images/{random.randint(0, 100000)}.jpg"
    with open(file_path, "wb") as f:
        f.write(image)
    
    image = tf.keras.preprocessing.image.load_img(file_path, target_size=(300, 300))

    # Convert the image to a numpy array
    image = tf.keras.preprocessing.image.img_to_array(image, dtype=np.uint8)
    
    # Resize the image
    image = np.array(image) / 255.0
    
    # Get the prediction
    prediction = model.predict(image[np.newaxis, ...]) # type: ignore
    
    # Get the label
    predicted_class = labels[np.argmax(prediction)]
    
    # Return the label
    return {"prediction": predicted_class}