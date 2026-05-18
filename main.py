from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import tensorflow as tf
import numpy as np
import json
import uvicorn
import traceback
import os

from PIL import Image


# -----------------------------------
# CREATE FASTAPI APP
# -----------------------------------

app = FastAPI()


# -----------------------------------
# ENABLE CORS
# -----------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------
# GET CURRENT DIRECTORY
# -----------------------------------

script_dir = os.path.dirname(
    os.path.abspath(__file__)
)


# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = None
class_names = None


model_path = os.path.join(script_dir, "Brain_Tumor_Detection_Final_Model.h5")

print("Looking for model at:", model_path)
print("Directory files:", os.listdir(script_dir))

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at: {model_path}")

model = tf.keras.models.load_model(model_path, compile=False)
print("Model Loaded Successfully")


# -----------------------------------
# LOAD CLASS NAMES
# -----------------------------------

try:

    class_path = os.path.join(
        script_dir,
        "Final_class_names.json"
    )

    if not os.path.exists(class_path):

        raise FileNotFoundError(
            f"Class names file not found: {class_path}"
        )

    with open(class_path, "r") as f:

        class_names = json.load(f)

    print("Class Names Loaded Successfully")

except Exception as e:

    print("Class Names Loading Error")

    print(e)


# -----------------------------------
# IMAGE PREPROCESSING FUNCTION
# -----------------------------------

def preprocess_image(image_file):

    try:

        image = Image.open(image_file)

        image = image.convert("RGB")

        image = image.resize((224, 224))

        image = np.array(image)

        image = tf.keras.applications.efficientnet.preprocess_input(image)

        image = np.expand_dims(image,axis=0)

        return image

    except Exception as e:

        raise Exception(
            f"Image Preprocessing Error: {str(e)}"
        )


# -----------------------------------
# HOME ROUTE
# -----------------------------------

@app.get("/")

def home():

    return {

        "message": "Brain Tumor Detection API Running",

        "status": "Backend Running Successfully"
    }


# -----------------------------------
# PREDICTION ROUTE
# -----------------------------------

@app.post("/predict")

async def predict(file: UploadFile = File(...)):

    if model is None or class_names is None:
        return JSONResponse(status_code=500,content={"error": "Model not loaded properly"})
  

    try:

        

        # check file uploaded
        if file is None:

            return JSONResponse(
                status_code=400,
                content={
                    "error": "No file uploaded"
                }
            )

        # validate extension
        allowed_extensions = [
            ".jpg",
            ".jpeg",
            ".png"
        ]

        file_extension = (
            "." + file.filename.lower().split(".")[-1]
        )

        if file_extension not in allowed_extensions:

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Only JPG, JPEG and PNG files are allowed"
                }
            )

        # preprocess image
        image = preprocess_image(
            file.file
        )

        # predict
        predictions = model.predict(image)

        predicted_index = np.argmax(
            predictions[0]
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = float(
            np.max(predictions)
        )

        confidence_percentage = round(
            confidence * 100,
            2
        )

        return {

            "prediction": predicted_class,

            "confidence": confidence_percentage
        }

    except Exception as e:

        print("Prediction Error")

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


# -----------------------------------
# GLOBAL ERROR HANDLER
# -----------------------------------

@app.exception_handler(Exception)

async def global_exception_handler(
    request,
    exc
):

    print("Global Exception Occurred")

    traceback.print_exc()

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error"
        }
    )


# -----------------------------------
# RUN SERVER
# -----------------------------------

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.environ.get("PORT", 8000)
        )
    )