import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import urllib.request
import requests # For extracting images from urls
from io import BytesIO
from PIL import Image

import os
import requests

from model.util import *
from model.eccv16 import *
from model.siggraph17 import *

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

def load_from_st(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    return tfile.name


def main():
    st.title("Image Colorizer")
    st.subheader("Created by Just Caleb (Choe)")
    file_name = None

    uploaded_file = st.file_uploader("Upload Files", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        file_name = load_from_st(uploaded_file)
        img = load_img(file_name)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))

    st.markdown("<p style='text-align: center;'>OR</p>",
            unsafe_allow_html=True)

    image_url = st.text_input("URL: ")
    if file_name is None and image_url:
        try:
            response = requests.get(image_url)
            uploaded_file = BytesIO(response.content)
            file_name = load_from_st(uploaded_file)
            img = load_img(file_name)
            (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
        except:
            st.write("Please enter a valid URL")

    model = st.radio("Colorize with one of these two models: ",
                         ('ECCV16', 'SIGGRAPH17'))

    if st.button("Colorize!") and file_name is not None:
        with st.spinner("Doing some magic ~~~"):
            try:
                img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
                if model == "ECCV16":
                    colorized = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
                    st.balloons()
                else:
                    colorized = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
                st.write("Result:")
                first, second = st.columns(2)
                first.image(img_bw, channels="BGR")
                second.image(colorized, channels="BGR")
            except:
                st.write("Something went wrong. Sending kittens to fix it.")
    else:
        st.write("Please either upload an image or enter a URL to an image")

if __name__ == "__main__":
    main()
