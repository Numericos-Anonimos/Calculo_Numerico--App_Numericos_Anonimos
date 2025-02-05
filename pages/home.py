import pandas as pd
from datetime import datetime
import streamlit as st
from PIL import Image

im = Image.open("src/img/unifesp_icon.ico")
st.set_page_config(
    page_title="Numéricos Anônimos",
    page_icon=im,
    layout="wide",
)

st.title("Numéricos Anônimos")

st.logo(
    im,
    link="https://portal.unifesp.br/",
    icon_image=im,
)

col1, col2 = st.columns(2, gap='small')

with col1:
    with st.container(border=True):
        st.page_link("pages/polinomio.py", label="**Aproximação de Polinômio**", icon="🔢", use_container_width=True)
        st.caption("Aproxima polinômios etc")
    with st.container(border=True):
        st.page_link("pages/lineares.py", label="**Sistemas Lineares**", icon="📐", use_container_width=True)
        st.caption("Resolução Sistemas Lineares")
    with st.container(border=True):
        st.page_link("pages/interpolacao.py", label="**Interpolação etc**", icon="📊", use_container_width=True)
        st.caption("Interpolação etc")

with col2:
    with st.container(border=True):
        st.page_link("pages/derivacao.py", label="**Derivação**", icon="📉", use_container_width=True)
        st.caption("Derivar funções")
    with st.container(border=True):
        st.page_link("pages/integracao.py", label="**Integração**", icon="📈", use_container_width=True)
        st.caption("Integrar funções")

st.write("Descrição sobre o projeto etc")