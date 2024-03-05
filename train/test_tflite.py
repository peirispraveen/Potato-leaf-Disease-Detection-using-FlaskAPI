import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf


def prediction(img_path, model_path):
    class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the image
    my_image = load_img(img_path, target_size=(256, 256))
    my_image = img_to_array(my_image)
    my_image = np.expand_dims(my_image, 0)
    my_image = my_image.astype('float32') / 255

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], my_image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    prediction_output = interpreter.get_tensor(output_details[0]['index'])

    # Plot the predictions
    fig = plt.figure(figsize=(7, 4))
    plt.barh(class_names, prediction_output[0], color='lightgray', edgecolor='red', linewidth=1, height=0.5)

    for index, value in enumerate(prediction_output[0]):
        plt.text(value / 2 + 0.1, index, f"{100 * value:.2f}%", fontweight='bold')

    plt.xticks([])
    plt.yticks([0, 1, 2], labels=class_names, fontweight='bold', fontsize=14)
    plt.savefig('pred_img.png', bbox_inches='tight')
    plt.show()


# Path to the TFLite model
model_path = "../models/potato_model.tflite"

# Path to the image for prediction
img_path = "C:/Users/ASUS/Desktop/test img/pothealthy.JPG"

# Make predictions
prediction(img_path, model_path)
