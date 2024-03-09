import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub

def detect_rice_diseases(filename):

    # Define custom object dictionary for loading the model
    custom_objects = {'KerasLayer': hub.KerasLayer}

    # Load trained model
    model_path = 'E:/AIDS Y2/DSGP/riceModel.h5'
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    # Define your image path (replace with the actual path to your image)
    image_path = filename

    def load_image(filename):
        img = cv2.imread(filename)
        img = cv2.resize(img, (224, 224))
        img = img / 255
        return img

    # Load the image
    img = load_image(image_path)

    classes = ['Bacterial_leaf_blight', 'blast', 'brownspot', 'normal']

    # Make predictions
    probabilities = model.predict(np.asarray([img]))[0]
    class_idx = np.argmax(probabilities)
    predicted_class = classes[class_idx]
    confidence = probabilities[class_idx]

    print(f"PREDICTED class: {predicted_class}, confidence: {confidence:.4f}")

    # Plotting the predicted value percentage for each class
    plt.figure(figsize=(12, 6))
    bars = plt.barh(classes, probabilities * 100, color='skyblue')
    plt.xlabel('Probability')
    plt.ylabel('Classes')
    plt.title('Predicted Value Percentage for Each Class')
    plt.xlim(0, 100)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Annotate each bar with its probability value
    for bar, prob in zip(bars, probabilities * 100):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{prob:.2f}',
                 va='center', ha='left')

    pred_plot = "static/prediction_plots/predicted_plot.png"
    plt.savefig(pred_plot)

    return pred_plot
