import requests
import streamlit as st
from PIL import Image
import base64
import json

st.set_option("deprecation.showfileUploaderEncoding", False)
st.title("Visual Question Answering")

image= st.file_uploader("Choose an Image")
st.image(image)
ques=st.text_input("Ask the Question")

if st.button("compute"):
    if image is not None and ques is not None:
        form_data={
            'image_bytes': base64.b64encode(image.getvalue()).decode('utf-8'),
            'question': ques
        }
        res= requests.post('http://backend:80/inputs', data=form_data, timeout=40)
        print(res.json())

        ans=st.write(f"Predicted_answer is ***{res.json()}***")