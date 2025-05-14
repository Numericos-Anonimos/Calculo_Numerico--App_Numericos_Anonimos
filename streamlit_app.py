import streamlit as st

st.set_page_config(
    page_title="NumTutor",
    page_icon= "src/img/unifesp_icon.ico",
    layout="wide",
)

pg = st.navigation([
    st.Page("pages/[0] Home.py", title="Home", icon="🏠"),
    st.Page("pages/[1] Equações de uma Variável.py", title="Equações de uma Variável", icon="🔢"),
    st.Page("pages/[2] Sistemas Lineares.py", title="Sistemas Lineares", icon="📐"),
    #st.Page("pages/[3] Sistemas Não Lineares.py", title="Sistemas Não Lineares", icon="📏"),
    st.Page("pages/[4] Interpolação.py", title="Interpolação", icon="📊"),
    st.Page("pages/[5] Mínimos Quadrados.py", title="Mínimos Quadrados", icon="📉"),
    st.Page("pages/[6] Integração Numérica.py", title="Integração Numérica", icon="📈"),
    st.Page("pages/[7] Problemas de Valor Inicial.py", title="Problemas de Valor Inicial", icon="📝"),
])

if st.session_state.get("chat_history") is None:
    st.session_state["chat_history"] = []

with st.sidebar:
    import chat as chat
    chat.print_chat_history()

pg.run()
