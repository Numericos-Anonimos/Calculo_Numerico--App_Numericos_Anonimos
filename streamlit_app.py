import streamlit as st

pg = st.navigation([
    st.Page("pages/home.py", title="Home", icon="🏠"),
    st.Page("pages/polinomio.py", title="Equações de uma Variável", icon="🔢"),
    st.Page("pages/lineares.py", title="Sistemas Lineares", icon="📐"),
    st.Page("pages/interpolacao.py", title="Interpolação", icon="📊"),
    st.Page("pages/integracao.py", title="Integração Numérica", icon="📈"),
    st.Page("pages/derivacao.py", title="Derivada", icon="📝"),
])

pg.run()