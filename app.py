import tensorflow as tf
import numpy as np 
import streamlit as st

import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

model_path = r'C:\Users\laptop\Desktop\Projects\crop disease deep learning\model_vgg19.h5'

# loading the trained model

model = tf.keras.models.load_model(model_path)

class_indices = {
                    "0":"Early_Blight",
                    "1":"Healthy",
                    "2":"Late_Blight"
}

def load_and_preprocess_image(image_path,model):

    img = image.load_img(image_path,target_size=(224,224))
    img = image.img_to_array(img)
    img = img/255
    img =np.expand_dims(img,axis=0)
    img_data=preprocess_input(img)
    prediction = model.predict(img_data)
    class_index = np.argmax(model.predict(img_data), axis=1)
    return class_index



st.title("Crop Disease Identification")

uploaded_img = st.file_uploader("Upload an image...", type=["jpg",'jpeg',"png"])

if uploaded_img is not None:

    img = image.load_img(uploaded_img)
    col1, col2 = st.columns(2)
   
    with col1:
        crop_img = img.resize((150, 150))
        st.image(crop_img)

    with col2:
        if st.button('Predict'):
            prediction = load_and_preprocess_image(uploaded_img,model)  
            st.write(prediction)
        
            if(prediction==0):
               st.success('Early_Blight')
            elif(prediction==1):
               st.success('Healthy')
            elif(prediction==2):
               st.success('Late_Blight')
            else:
               print('Invalid')
    