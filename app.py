from flask import Flask, render_template, request
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os
import time

app = Flask(__name__)
app.static_folder = 'static'

# Load your pre-trained model
model = tf.keras.models.load_model("models/potato_mobilenetv2_model.h5")

# Define the upload directory
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        file = request.files['file']
        input_img = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], input_img))

        pred_img_path = predict(input_img)
        return render_template('index.html', input_img=input_img, pred_img=pred_img_path)
    return render_template('index.html')


def predict(input_img):
    class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

    # Load and preprocess the image
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], input_img)

    start_time = time.time()

    my_image = load_img(img_path, target_size=(256, 256))
    my_image = img_to_array(my_image)
    my_image = np.expand_dims(my_image, axis=0)
    my_image = my_image.astype('float32') / 255

    prediction_output = model.predict(my_image)

    # Plot the predictions
    fig = plt.figure(figsize=(7, 4))
    plt.barh(class_names, prediction_output[0], color='lightgray', edgecolor='red', linewidth=1, height=0.5)

    for index, value in enumerate(prediction_output[0]):
        plt.text(value / 2 + 0.1, index, f"{100 * value:.2f}%", fontweight='bold')

    plt.xticks([])
    plt.yticks([0, 1, 2], labels=class_names, fontweight='bold', fontsize=14)

    name = app.config['UPLOAD_FOLDER']+'pred_img.png'
    fig.savefig(name, bbox_inches='tight')

    end_time = time.time()

    time_taken = end_time - start_time
    print("time taken: ", time_taken)

    return name


if __name__ == '__main__':
    app.run(debug=True)
