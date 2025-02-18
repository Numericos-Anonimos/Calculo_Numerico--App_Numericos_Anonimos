import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

def baixar_arquivo(endereço):
    with open(endereço, "r", encoding="utf-8") as file:
        return file.read()

def buscar_contexto(page):
    match page:
        case "Home":
            return baixar_arquivo('resumos\Home.md')
        case "Equações de uma Variável":
            return '\n\n\n'.join([baixar_arquivo(i) for i in ['resumos\[ 1 ] Bisseção.md', 'resumos\[ 1 ] Heron Newton.md', 'resumos\[ 1 ] Método das Secantes.md']])
        case "Sistemas Lineares":
            return baixar_arquivo('resumos\[ 2 ] Jacobi e Gauss-Seidel.md')
        case "Sistemas Não Lineares":
            return baixar_arquivo('resumos\[ 3 ] Método de Newton (Não linear).md')
        case "Interpolação":
            return baixar_arquivo('resumos\[ 4 ] Interpolação.md')
        case "Mínimos Quadrados":
            return baixar_arquivo('resumos\[ 5 ] Mínimos Quadrados.md')
        case "Integração Numérica":
            return baixar_arquivo('resumos\[ 6 ] Integração.md')
        case "Problemas de Valor Inicial":
            return '\n\n\n'.join([baixar_arquivo(i) for i in ['resumos\[ 7 ] Método de Euler.md', 'resumos\[ 7 ] Runge Kutta.md']])
        case _:
            return "Página não encontrada"

import chatbot.chain as c
chain = c.create_chain(temperature=0.1)

def print_chat_history():
    with st.expander("Chat", expanded=True):
        for message in st.session_state["chat_history"]:
            if isinstance(message, AIMessage):
                with st.chat_message("Númerico Anonimo", avatar="https://em-content.zobj.net/source/animated-noto-color-emoji/356/robot_1f916.gif"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Você", avatar="https://em-content.zobj.net/source/animated-noto-color-emoji/356/smiling-face-with-sunglasses_1f60e.gif"):
                    st.write(message.content)

        user_input = st.chat_input("Digite sua pergunta aqui")

        if user_input:
            st.session_state["chat_history"].append(HumanMessage(user_input))
            with st.chat_message("Você", avatar="https://em-content.zobj.net/source/animated-noto-color-emoji/356/smiling-face-with-sunglasses_1f60e.gif"):
                st.write(user_input)
            with st.chat_message("Númerico Anonimo", avatar="https://em-content.zobj.net/source/animated-noto-color-emoji/356/robot_1f916.gif"):
                resposta = st.write_stream(c.talk(chain, user_input, buscar_contexto(st.session_state['current_page']), st.session_state["chat_history"][-5:]))
                st.session_state["chat_history"].append(AIMessage(resposta))
                st.rerun()
