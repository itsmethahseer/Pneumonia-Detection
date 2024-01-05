
# Pneumonia Detection Project

## Overview

This repository contains the code for a Pneumonia Detection model using a pre-trained ResNet152V2 deep learning model. The project is implemented using FastAPI to expose an API for real-time predictions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/itsmethahseer/Pneumonia-Detection.git
   cd Pneumonia-Detection
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained ResNet152V2 model weights and place them in the project directory. You can find the model weights by running the jupyter notebook.

4. Run the FastAPI app:

   ```bash
   uvicorn main:app --reload
   ```

## Usage

Once the FastAPI app is running, you can use the provided API to make predictions. Visit the Swagger UI at `http://127.0.0.1:8000/docs` for interactive documentation. Use the `/predict` endpoint to upload an image and receive predictions.

Example Usage:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "image=@path/to/your/image.jpg"
```

## Directory Structure

```
pneumonia-detection/
|-- main.py               # FastAPI application code
|-- .gitignore    
|-- requirements.txt     # Project dependencies
|-- README.md            # Project README file
|-- pneumonia-detection.ipynb         # Model creating notbook
```

## Dependencies

- TensorFlow
- FastAPI
- Keras
- NumPy
- Matplotlib
- Seaborn
- Other dependencies as specified in `requirements.txt`

## Acknowledgments

- [Link to the dataset source](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)
- Mention any other external resources or references



Feel free to customize the sections according to your specific project details. Include relevant links, such as the source of your dataset, model weights, or any external resources you utilized. Providing clear and comprehensive information in your README enhances the user experience and encourages contributions from the community.
