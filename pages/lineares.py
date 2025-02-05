import streamlit as st
from st_mathlive import mathfield
from PIL import Image

im = Image.open("src/img/unifesp_icon.ico")

st.logo(
    im,
    link="https://portal.unifesp.br/",
    icon_image=im,
)

st.title("Sistemas Lineares")