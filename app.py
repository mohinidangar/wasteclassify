from flask import Flask, request, render_template, jsonify
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the pre-trained AI model
model = load_model('waste_classifier_finetuned_mobilenetv2.h5')

# Define the path to save uploaded images
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Route to upload image and get prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Preprocess image
        img = image.load_img(filename, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Get prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)

        # Map the predicted class index to class name
        classes = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic',
                   'shoes', 'trash', 'white-glass']
        predicted_label = classes[predicted_class[0]]

        # Return the predicted label as a response
        return jsonify({'predicted_class': predicted_label})

    return jsonify({'message': 'Invalid file format'})


# Home route
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
