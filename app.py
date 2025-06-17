import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model_cnn = tf.keras.models.load_model('weight_test.h5')
model_resnet = tf.keras.models.load_model('resnet_weight.h5')
model_inception = tf.keras.models.load_model('inception_weight.h5')
model_alexnet = tf.keras.models.load_model('densenet_weight.h5')

class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def make_prediction(model, image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
   
    img_array = img_array / 255.0
    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    class_label = class_names[class_index]
    confidence = np.max(pred) * 100
    return class_label, confidence

st.title('Dementia Classification')
st.markdown("Upload an image and select a model to classify the dementia type.")

file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])

if file_uploaded is not None:
    image = Image.open(file_uploaded)
    image.thumbnail((400, 400))  
    st.image(image, caption='Uploaded Image')

model_selection = st.radio("Select Model", ('CNN', 'ResNet', 'Inception', 'DenseNet'))

class_btn = st.button("Check")

if class_btn:
    if file_uploaded is None:
        st.write("Invalid command, please upload an image")
    else:
        with st.spinner('Model working....'):
            if model_selection == 'CNN':
                model = model_cnn
            elif model_selection == 'ResNet':
                model = model_resnet
            elif model_selection == 'Inception':
                model = model_inception
            elif model_selection == 'DenseNet':
                model = model_alexnet

            class_label, confidence = make_prediction(model, image)
            st.success('Prediction Successful.')
            st.write("Predicted Image:", class_label)
            st.write("Confidence:", f"{confidence:.2f}%")

comparative_analysis_btn = st.button("Comparative Analysis")

if comparative_analysis_btn:
    if file_uploaded is None:
        st.write("Invalid command, please upload an image")
    else:
        with st.spinner('Performing Comparative Analysis....'):
            # Make predictions for all models
            predictions_cnn = make_prediction(model_cnn, image)
            predictions_resnet = make_prediction(model_resnet, image)
            predictions_inception = make_prediction(model_inception, image)
            predictions_alexnet = make_prediction(model_alexnet, image)

            # Plot the comparative graph
            fig, ax = plt.subplots(figsize=(8, 6))  # Specify the desired figure size here
            models = ['CNN', 'ResNet', 'Inception', 'DenseNet']
            confidence = [
                predictions_cnn[1],
                predictions_resnet[1],
                predictions_inception[1],
                predictions_alexnet[1]
            ]
            ax.bar(models, confidence)
            ax.set_xlabel('Models')
            ax.set_ylabel('Confidence')
            ax.set_title('Comparative Analysis')

            for i in range(len(models)):
                ax.text(models[i], confidence[i], f"{confidence[i]:.2f}%", ha='center', va='bottom')

            st.write("Predicted Image:", predictions_cnn[0])
            st.write("Confidence:", f"{predictions_cnn[1]:.2f}%")

            st.pyplot(fig)
