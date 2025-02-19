import streamlit as st
import pandas as pd
from st_mathlive import mathfield
from PIL import Image
import numpy as np
import plotly.graph_objects as go

st.session_state['current_page'] = "Interpolação"

st.logo(
    "src/img/unifesp_icon.ico",
    link="https://portal.unifesp.br/",
    icon_image="src/img/unifesp_icon.ico",
)

#Newton
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


#Lagrange
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


def plot_polinomio_interpolador(coef, x_pontos, y_pontos, epsilon=1e-10):
    # Calcula os coeficientes do polinômio interpolador
    # Define a função polinomial com os coeficientes obtidos.
    # Os coeficientes estão em ordem decrescente:
    # p(x) = coef[0]*x^(n-1) + coef[1]*x^(n-2) + ... + coef[n-1]
    def polynomial(x):
        y = 0
        n_coef = len(coef)
        for i, c in enumerate(coef):
            y += c * (x ** (n_coef - i - 1))
        return y
    
    # Define o intervalo para plotar a curva
    x_pontos = np.array(x_pontos, dtype=float)
    x_min, x_max = x_pontos.min() - 1, x_pontos.max() + 1
    x_range = np.linspace(x_min, x_max, 400)
    y_range = [polynomial(x) for x in x_range]
    
    # Cria DataFrames para os pontos originais e para a curva do polinômio
    df_pontos = pd.DataFrame({"x": x_pontos, "y": y_pontos})
    df_polynomial = pd.DataFrame({"x": x_range, "y": y_range})
    
    # Cria o gráfico com os pontos originais
    fig = px.scatter(
        df_pontos, x="x", y="y",
        title="Polinômio Interpolador",
        labels={"x": "x", "y": "y"},
        template="plotly_dark",
        color_discrete_sequence=["cyan"]
    )
    
    # Adiciona a linha da curva do polinômio interpolador
    fig.add_traces(px.line(
        df_polynomial, x="x", y="y",
        template="plotly_dark",
        color_discrete_sequence=["yellow"]
    ).data)
    
    return fig


def apresentando_interpolacao():
    graf = go.figure()
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
        st.title("Interpolação Lagrange: ")

        # Extrair os pontos x e y e converter para float
        x_pontos = df['X'].astype(float).values
        y_pontos = df['y'].astype(float).values

        # Verificar se todos os valores são NaN
        if np.isnan(x_pontos).all() or np.isnan(y_pontos).all():
            st.error("Não foi possível realizar a interpolação. A tabela contém apenas valores vazios.")
        else:
            # Calcular o polinômio interpolador
            coef = polinomio_interpolador(x_pontos, y_pontos)
            graf = plot_polinomio_interpolador(coef, x_pontos, y_pontos)
            st.write(f"Coeficientes do polinômio interpolador: {coef}")

            # Exibir o polinômio resultante
            polinomio = "f(x) = "
            for i, c in enumerate(coef):
                polinomio += f"{c:.4f}x^{len(coef) - i - 1} "
                if i < len(coef) - 1:
                    polinomio += "+ "
            st.write(f"Polinômio interpolador: {polinomio}")

            st.write("Método incremental que usa diferenças divididas para construir o polinômio. Permite adicionar pontos sem recalcular os anteriores, sendo mais eficiente para novos dados.")

            st.plotly_chart(graf)

    elif "newton" in st.session_state and st.session_state["newton"] and "Dados" in st.session_state:
        st.markdown("---")
        st.title("Interpolação Newton: ")

        # Extrair os pontos x e y e converter para float
        x_pontos = df['X'].astype(float).values
        y_pontos = df['y'].astype(float).values

        # Verificar se todos os valores são NaN
        if np.isnan(x_pontos).all() or np.isnan(y_pontos).all():
            st.error("Não foi possível realizar a interpolação. A tabela contém apenas valores vazios.")
        else:
            # Calcular os coeficientes do polinômio de Newton
            coef = polinomio_newton(x_pontos, y_pontos)
            st.write(f"Coeficientes do polinômio de Newton: {coef}")

            # Construir a string do polinômio
            polinomio_str = f"{coef[0]:.4f}"
            for i in range(1, len(coef)):
                termo = " * ".join([f"(x - {x_pontos[j]:.4f})" for j in range(i)])
                polinomio_str += f" + ({coef[i]:.4f}) * {termo}"

            st.write(f"Polinômio interpolador de Newton: {polinomio_str}")

            st.write("Método que encontra um polinômio que passa exatamente por todos os pontos dados, usando uma combinação de funções base. É eficiente, mas pode ser instável para grandes conjuntos de dados.")



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
            if st.button("Interpolador de Newton"):
                if not df.empty:
                    st.session_state["Dados"] = df
                    st.session_state["newton"] = True
                    st.empty()
                    st.rerun()
            st.write("Método que encontra um polinômio que passa exatamente por todos os pontos dados, usando uma combinação de funções base. É eficiente, mas pode ser instável para grandes conjuntos de dados.")

    # Verificar se o método foi selecionado anteriormente
    

apresentando_interpolacao()