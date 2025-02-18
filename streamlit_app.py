import streamlit as st

pg = st.navigation([
    st.Page("pages/home.py", title="Home", icon="🏠"),
    st.Page("pages/polinomio.py", title="Equações de uma Variável", icon="🔢"),
    st.Page("pages/lineares.py", title="Sistemas Lineares", icon="📐"),
    #st.Page("pages/naolineares.py", title="Sistemas Não Lineares", icon="📏"),
    st.Page("pages/interpolacao.py", title="Interpolação", icon="📊"),
    #st.Page("pages/ajuste.py", title="Mínimos Quadrados", icon="📉"),
    #st.Page("pages/integracao.py", title="Integração Numérica", icon="📈"),
    st.Page("pages/derivacao.py", title="Problemas de Valor Inicial", icon="📝"),
])

if st.session_state.get("chat_history") is None:
    st.session_state["chat_history"] = []

with st.sidebar:
    import chatbot.chat as chat
    chat.print_chat_history()

pg.run()
