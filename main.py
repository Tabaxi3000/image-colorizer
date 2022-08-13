import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import urllib.request
import requests # For extracting images from urls
from io import BytesIO

def load_network():
    prototxt = r"model/colorization_deploy_v2.prototxt"
    model = r"model/colorization_release_v2.caffemodel"
    points = r"model/pts_in_hull.npy"
    points = os.path.join(os.path.dirname(__file__), points)
    prototxt = os.path.join(os.path.dirname(__file__), prototxt)
    model = os.path.join(os.path.dirname(__file__), model)
    net = cv2.dnn.readNetFromCaffe(prototxt, model)  # load model from disk
    pts = np.load(points)

    # add the cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net

def colorize(file_name, net):
    image = cv2.imread(file_name)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # resize the Lab image to 224x224 (the dimensions the colorization network accepts),
    # split channels, extract the 'L' channel, and then perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # pass the L channel through the network which will *predict* the 'a' and 'b' channel values
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # resize the predicted 'ab' volume to the same dimensions as our input image
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # grab the 'L' channel from the *original* input image
    # (not the resized one) and concatenate the original 'L' channel with the predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # convert the output image from the Lab color space to RGB,
    # then clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # the current colorized image is represented as a floating point
    # data type in the range [0, 1] -- let's convert to an unsigned 8-bit integer representation in the range [0, 255]
    colorized = (255 * colorized).astype("uint8")
    return image, colorized


def load_from_st(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    return tfile.name


def main():
    net = load_network()
    st.title("Image Colorizer")
    st.subheader("Created by Just Caleb (Choe)")
    file_name = None

    uploaded_file = st.file_uploader("Upload Files", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        file_name = load_from_st(uploaded_file)

    st.markdown("<p style='text-align: center;'>OR</p>",
            unsafe_allow_html=True)

    image_url = st.text_input("URL: ")
    if file_name is None and image_url:
        try:
            response = requests.get(image_url)
            uploaded_file = BytesIO(response.content)
            file_name = load_from_st(uploaded_file)
        except:
            st.write("Please enter a valid URL")

    if st.button("Colorize!") and file_name is not None:
        with st.spinner("Doing some magic ~~~"):
            image, colorized = colorize(file_name, net)
            st.balloons()
            st.write("Result:")
            first, second = st.columns(2)
            first.image(image, channels="BGR")
            second.image(colorized, channels="BGR")


if __name__ == "__main__":
    main()
