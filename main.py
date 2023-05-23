"""
Bottle Classifier API

This API has one endpoint, /bottle_classify, which takes an image file as input and returns a prediction of whether 
the bottle is accepted or rejected.

Also save the images uploaded to the API to the disk for future use.
"""
import tensorflow as tf
import numpy as np
import random
import firebase_admin
from fastapi import FastAPI, File, UploadFile
from firebase_admin import credentials, firestore


# Constants
FIREBASE_CONFIG = "config/refil-v1-firebase-adminsdk-bqu2h-e79deef49e.json"
LABELS = ["Accepted", "Rejected"]
REJECTED_POINTS = 25

# Initialize the app
app = FastAPI()
model = tf.keras.models.load_model("config/model_14_0.93.h5")
cred = credentials.Certificate(FIREBASE_CONFIG)
firebase = firebase_admin.initialize_app(cred)
firestore_db = firestore.client(firebase)
user_collection = firestore_db.collection("users")
bottles_collection = firestore_db.collection("bottles")


# Routes
@app.post("/api/bottle-classify")
async def bottle_classify(file: UploadFile = File(...)) -> dict:
    # Read the image
    image: bytes = await file.read()

    # Save the image to disk
    file_path = f"images/{random.randint(0, 100000)}.jpg"
    with open(file_path, "wb") as f:
        f.write(image)

    image_processed = tf.keras.preprocessing.image.load_img(
        file_path, target_size=(300, 300)
    )

    # Convert the image to a numpy array
    image_array = tf.keras.preprocessing.image.img_to_array(
        image_processed, dtype=np.uint8
    )

    # Resize the image
    image_resized = np.array(image_array) / 255.0

    # Get the prediction
    prediction = model.predict(image_resized[np.newaxis, ...])  # type: ignore

    # Get the label
    predicted_class = LABELS[np.argmax(prediction)]

    # Return the label
    return {"prediction": predicted_class}


@app.get("/api/user")
async def get_user(uuid: str):
    user_document: dict = user_collection.document(uuid).get().to_dict()
    user_response: dict = {
        "uuid": uuid,
        "email": user_document["email"],
        "name": user_document["name"],
        "points": user_document["points"],
    }
    return user_response


@app.post("/api/bottle")
async def process_bottle(uuid: str, barcode: str, image_prediction: str):
    # Get the user document
    user_document: dict = user_collection.document(uuid).get().to_dict()

    # Get the bottle document
    bottle_document: dict = bottles_collection.document(barcode).get().to_dict()

    # Set the points to add
    points_to_add: int = (
        bottle_document["points"] if image_prediction == "Accepted" else REJECTED_POINTS
    )

    # Update the user document
    user_document["points"] += points_to_add
    user_document["bottles"] += 1
    user_collection.document(uuid).set(user_document)

    print(
        f"User {uuid} has scanned bottle {barcode} and the image prediction is {image_prediction}"
    )
