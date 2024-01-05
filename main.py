from fastapi import FastAPI
from fastapi import UploadFile,File
import tensorflow as tf
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.resnet_v2 import preprocess_input
import keras as k
import numpy as np
from tempfile import NamedTemporaryFile
app = FastAPI()

model = k.models.load_model("/home/c847/Desktop/Pneumonia Detection/finetuned_from_resnet/pretrainedmodel_fromResnet.h5")

IMG_SIZE = 224
model_threshold = 0.5 
def preprocess_image(image_path):
    # Load and resize the image
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    
    # Expand the dimensions to match the model's expected format
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image for the ResNet152V2 model
    img_array = preprocess_input(img_array)
    
    return img_array

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    # Create a temporary file to store the uploaded content
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(upload_file.file.read())
    return tmp_file.name



def classify_pneumonia(prediction):
    return "Pneumonia" if prediction >= model_threshold else "Not Pneumonia"

@app.get("/")
async def Hello():
    return {"message":"Fastapi is initialized"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        tmp_file_path = save_upload_file_tmp(image)

        # Preprocess the image using the file path
        img_array = preprocess_image(tmp_file_path)

        # Make a prediction using your model
        prediction = model.predict(img_array)[0][0]

        # Convert NumPy float32 to Python float
        prediction = float(prediction)

        # Classify based on the threshold
        classification = classify_pneumonia(prediction)

        # Return the result as JSON
        return {"prediction": prediction, "classification": classification}

    except Exception as e:
        return {"error": str(e)}


