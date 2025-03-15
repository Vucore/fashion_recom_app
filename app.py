import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
import streamlit as st
import gdown
st.header('Fashion Recommendation System')

filenames_path = "filenames.pkl"
image_feature_data_path = "image_features.pkl"

if not os.path.exists(filenames_path):
    gdown.download("https://drive.google.com/uc?id=1bWoBSU5Hq1wNQUYuP_BS565P4C3Rr7Bp", filenames_path, quiet=False)
if not os.path.exists(image_feature_data_path):
    gdown.download("https://drive.google.com/uc?id=1iq7xKxz_LUZDIT0wCix0fIr4KmrjgjK1", image_feature_data_path, quiet=False)  

@st.cache_data
def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)
    
if os.path.exists(filenames_path) and os.path.exists(image_feature_data_path):
    try:
        filenames = load_pkl_file(filenames_path)
        image_feature_data = load_pkl_file(image_feature_data_path)
    except Exception as e:
        st.error(f"Error loading .pkl files: {str(e)}")
        st.stop()
    
# filenames = pkl.load(open(filenames_path, 'rb'))
# image_feature_data = pkl.load(open(image_feature_data_path, 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPool2D()
])

neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
neighbors.fit(image_feature_data)

def extract_features_from_images(image_path, model):
  img = image.load_img(image_path, target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_expand_dim = np.expand_dims(img_array, axis=0)
  img_preprocess = preprocess_input(img_expand_dim)
  result = model.predict(img_preprocess).flatten()
  norm_result = result / np.linalg.norm(result)
  return norm_result

upload_file = st.file_uploader('Upload Image')

if upload_file is not None:
    with open(os.path.join('upload', upload_file.name), 'wb') as f:
        f.write(upload_file.getbuffer())
    st.subheader('Uploaded Image')
    st.image(upload_file)
    input_image_features = extract_features_from_images(upload_file, model)
    distance, indices = neighbors.kneighbors([input_image_features])
    st.subheader('Recommend Images')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
       st.image(filenames[indices[0][1]])
    with col2:
       st.image(filenames[indices[0][2]])
    with col3:
       st.image(filenames[indices[0][3]])
    with col4:
       st.image(filenames[indices[0][4]])
    with col5:
       st.image(filenames[indices[0][5]])