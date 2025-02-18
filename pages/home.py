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

st.markdown("---")

st.header("Descrição do Projeto: ")

st.write("""
O projeto **Numérico Anônimos** foi desenvolvido com o objetivo de consolidar os conceitos adquiridos durante o semestre de Cálculo Numérico, ao mesmo tempo em que oferece uma ferramenta prática para a aplicação dos métodos matemáticos aprendidos. Este projeto visa ser uma contribuição útil e acessível para estudantes e profissionais que desejam entender e aplicar os métodos do Cálculo Numérico de forma simples e interativa.

### Objetivos

- Criar uma ferramenta intuitiva e prática para resolver problemas de Cálculo Numérico.
- Aplicar métodos fundamentais como **sistemas lineares**, **aproximação de raízes polinomiais**, **integração** e **derivação**.
- Fornecer uma solução acessível, especialmente para quem está aprendendo Cálculo Numérico.

### Tecnologias Utilizadas

- **Python**: Linguagem principal para o desenvolvimento da aplicação.
- **Bibliotecas Python**: Uso de bibliotecas como `Numpy`, `Pandas` e `Matplotlib` para implementação dos métodos matemáticos e visualização dos resultados.
- **Streamlit**: Framework utilizado para criar o site interativo.

### Funcionalidades

- **Métodos Matemáticos**: A aplicação oferece a implementação dos principais métodos de Cálculo Numérico, incluindo:
  
- **Interface Interativa**: Utilização de uma interface simples e clara para facilitar a interação do usuário com a ferramenta.

### Contribuições e Impacto

O projeto não apenas serviu para consolidar os conhecimentos adquiridos ao longo do semestre, mas também propõe uma ferramenta útil para qualquer pessoa que esteja interessada em aprender e aplicar os conceitos de Cálculo Numérico. A combinação de uma interface amigável com a funcionalidade do chatbot torna a experiência de aprendizado mais dinâmica e interativa.

O **Numérico Anônimos** representa uma forma prática e acessível de aplicar métodos matemáticos fundamentais, promovendo o entendimento e o uso desses conceitos em problemas reais e desafiadores.
""")
