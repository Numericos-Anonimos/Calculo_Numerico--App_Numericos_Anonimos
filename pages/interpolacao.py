import streamlit as st
import pandas as pd
from st_mathlive import mathfield
from PIL import Image

im = Image.open("src/img/unifesp_icon.ico")

st.logo(
    im,
    link="https://portal.unifesp.br/",
    icon_image=im,
)

st.title("Interpolação")

df = pd.DataFrame(columns=['X', 'y'])

df = st.data_editor(df, num_rows="dynamic")