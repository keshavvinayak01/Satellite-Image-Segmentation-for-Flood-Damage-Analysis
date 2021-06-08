import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from patch_inference import get_prediction, get_model
import time
# Call get_model in a cached manner before proceeding to the prediction

# get_model()
st.title('Image Segmentation App')


"## Upload a Satellite Image"
uploaded_file = st.file_uploader("Choose an image..", 
    type = ["jpg", "jpeg", "png"])

# Pass the path / virtual path of the image here


# mask, damage = get_prediction(uploaded_file)
# if mask and damage:
#     st.image([mask, damage], 
#         caption = ["Building Mask Prediction", "Flood Damage Mask Prediction"])

############# Testing code ################### 
if uploaded_file:
    with st.spinner("Prediction in progress..."):
    # Change this spinner to run until prediction task is completed
        time.sleep(5)
    st.balloons()
    st.success("Prediction Completed")
    
    
    "# Given Satellite Image: "
    st.image(uploaded_file, width = 600)
    "# Output:"
    st.image(["image1.jpeg", "image2.jpg"],
        caption = ["Building Mask Prediction", "Flood Damage Mask Prediction"])