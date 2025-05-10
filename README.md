# wasteclassify
# Waste Classifier Web App 

This is a Flask-based web application that uses a fine-tuned **MobileNetV2** deep learning model to classify uploaded waste images into one of 12 categories. The goal is to support eco-friendly waste sorting and recycling efforts by providing real-time waste type identification.

## Project Overview

The app allows users to:
- Upload an image of a waste item (e.g., battery, glass, plastic, etc.)
- Automatically classify the image into one of the following categories:
  - `battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass`
- View the predicted waste type instantly in the browser

This can assist recycling facilities or environmental platforms in efficiently identifying and sorting waste.

---

## AI Model

The model used is a **fine-tuned MobileNetV2**, trained on a custom dataset of labeled waste images. The image is resized, normalized, and passed through the model to predict the category.

---

## Requirements

To run this project, install the following dependencies:

```bash
pip install -r requirements.txt
Flask==2.3.2
tensorflow==2.13.0
numpy==1.24.3
Pillow==9.5.0

