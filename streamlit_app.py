import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.applications.imagenet_utils import decode_predictions

st.write("""
# Garbage Classification
"""
)

upload_file = st.sidebar.file_uploader("Upload Garbage Image", type=["jpg","png"])
Generate_pred=st.sidebar.button("Predict")
model=tf.keras.models.load_model('GarbageClassificationModel.h5')

def import_n_pred(image_data, model):
    size = (512,384)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape=img[np.newaxis,...]
    pred = model.predict(reshape)
    return pred

if Generate_pred:
    image=Image.open(upload_file)
    with st.expander('Garbage Image', expanded = True):
        st.image(image, use_column_width=True)
    pred=import_n_pred(image, model)
    labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    st.title("Prediction of image is {}".format(labels[np.argmax(pred)]))