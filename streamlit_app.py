import streamlit as st

pg = st.navigation([
    st.Page("pages/home.py", title="Home", icon="🏠"),
    st.Page("pages/polinomio.py", title="Aproximar polinômio", icon="🔢"),
    st.Page("pages/lineares.py", title="Sistemas Lineares", icon="📐"),
    st.Page("pages/interpolacao.py", title="Interpolação", icon="📊"),
    st.Page("pages/integracao.py", title="Integração", icon="📈"),
    st.Page("pages/derivacao.py", title="Derivação", icon="📉"),
])

pg.run()