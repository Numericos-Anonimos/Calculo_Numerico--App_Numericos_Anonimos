import streamlit as st
from st_mathlive import mathfield
from PIL import Image

def print_math():
    st.warning(Tex)
    sent = False

im = Image.open("src/img/unifesp_icon.ico")

st.logo(
    im,
    link="https://portal.unifesp.br/",
    icon_image=im,
)

st.title("Derivação")

Tex, MathML = mathfield(title="Equação para Derivar", value=r"",)

#st.latex(Tex)
#st.write(MathML)
st.button(label="Enviar", on_click=print_math)