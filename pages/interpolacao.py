import streamlit as st
import pandas as pd
from st_mathlive import mathfield
from PIL import Image
import numpy as np
import plotly.graph_objects as go



im = Image.open("src/img/unifesp_icon.ico")

st.logo(
    im,
    link="https://portal.unifesp.br/",
    icon_image=im,
)


def newton_polynomial(x, x_pontos, coef):
    n = len(coef)
    result = coef[0]
    for i in range(1, n):
        term = coef[i]
        for j in range(i):
            term *= (x - x_pontos[j])
        result += term
    return result

def diferenca(x, y):
    n = len(x)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        for i in range(n - j):
            coef[i] = (coef[i + 1] - coef[i]) / (x[i + j] - x[i])
    return coef[:n]

def polinomio_newton(x_points, y_points):
    coef = diferenca(x_points, y_points)



    polinomio_str = f"{coef[0]:.4f}"
    for i in range(1, len(coef)):
        termo = " * ".join([f"(x - {x_points[j]:.4f})" for j in range(i)])
        polinomio_str += f" + ({coef[i]:.4f}) * {termo}"

    return coef



def polynomial(x, coef):
    result = 0
    degree = len(coef) - 1
    for i in range(len(coef)):
        result += coef[i] * x ** (degree - i)
    return result

def lagrange_interpolation(x_pontos, y_pontos):
    n = len(x_pontos)
    result = 0
    for i in range(n):
        term = y_pontos[i]
        for j in range(n):
            if i != j:
                term *= (x_pontos[i] - x_pontos[j])
        result += term
    return result

def polinomio_interpolador(x_pontos, y_pontos, epsilon=1e-10):
    x_pontos = np.array(x_pontos, dtype=float)  # Garantir que x_pontos seja um array de floats
    y_pontos = np.array(y_pontos, dtype=float)  # Garantir que y_pontos seja um array de floats
    n = len(x_pontos)
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            A[i, j] = x_pontos[i] ** (n - j - 1)

    coef = np.linalg.solve(A, y_pontos)
    coef = np.array([0 if abs(c) < epsilon else c for c in coef])

    coef_arredondados = [round(c, 4) for c in coef]

    return coef_arredondados




def apresentando_interpolacao():
    st.title("Interpolação")
    st.markdown("\n\n\n\n")
    
    st.markdown("<small>Preencha a tabela a seguir com os pontos:</small>", unsafe_allow_html=True)

    # Criando as colunas para os botões ficarem na mesma linha
    col1, col2 = st.columns([1, 1])

    with col1:
        # Criação do DataFrame com as colunas 'X' e 'y'
        df = pd.DataFrame(columns=['X', 'y'])

        # Exibindo a tabela centralizada
        st.markdown(
            """
            <style>
            .stDataFrame {
                display: table;
                margin-left: auto;
                margin-right: auto;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Exibindo a tabela
        df = st.data_editor(df, num_rows="dynamic")

    if "lagrange" in st.session_state and st.session_state["lagrange"] and "Dados" in st.session_state:
        st.markdown("---")
        st.header("Polinômio Interpolador Lagrange: ")
        # Extrair os pontos x e y
        x_pontos = df['X'].values
        y_pontos = df['y'].values
                    
        # Calcular o polinômio interpolador
        coef = polinomio_interpolador(x_pontos, y_pontos)
        st.write(f"Coeficientes do polinômio interpolador: {coef}")
                    
                    # Exibir o polinômio resultante
        polinomio = "f(x) = "

        for i, c in enumerate(coef):
            polinomio += f"{c:.4f}x^{len(coef) - i - 1} "
            if i < len(coef) - 1:
                polinomio += "+ "
            st.write(f"Polinômio interpolador: {polinomio}")

        

        coef = polinomio_interpolador(x_pontos, y_pontos)
        st.write(coef)

    elif "newton" in st.session_state and st.session_state["newton"]:
        pass



    with col2:
        # Adicionando o método Lagrange
        with st.container(border=True):
            if st.button("Interpolador de Lagrange"):
                if not df.empty:
                    st.session_state["Dados"] = df
                    st.session_state["lagrange"] = True
                    st.empty()
                    st.rerun()

            st.write("Método incremental que usa diferenças divididas para construir o polinômio. Permite adicionar pontos sem recalcular os anteriores, sendo mais eficiente para novos dados.")

        # Adicionando o método Newton
        with st.container(border=True):
            if st.button("Interpolador de Newton") and not st.session_state["newton"]:
                if not df.empty:
                    st.session_state["Dados"] = df
                    st.session_state["newton"] = True
                    st.empty()
                    st.rerun()
            st.write("Método que encontra um polinômio que passa exatamente por todos os pontos dados, usando uma combinação de funções base. É eficiente, mas pode ser instável para grandes conjuntos de dados.")

    # Verificar se o método foi selecionado anteriormente
    

apresentando_interpolacao()