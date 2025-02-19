import streamlit as st
import pandas as pd
from st_mathlive import mathfield
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.session_state['current_page'] = "Interpolação"

#Newton
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def newton_polynomial(x, x_points, coef):
    x_points = np.array(x_points, dtype=float)
    result = float(coef[0])
    term = 1.0
    for i in range(1, len(coef)):
        term *= (x - float(x_points[i-1]))
        result += float(coef[i]) * term
    return result

def diferenca(x, y):    
    n = len(x)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])
    return coef

def polinomio_newton(x_points, y_points):
    coef = diferenca(x_points, y_points)
    coef = np.array(coef, dtype=float)
    
    polinomio_str = f"{float(coef[0]):.4f}"
    for i in range(1, len(coef)):
        termo = " * ".join([f"(x - {float(x_points[j]):.4f})" for j in range(i)])
        polinomio_str += f" + ({float(coef[i]):.4f}) * {termo}"
    
    return coef, polinomio_str

def plot_newton_interpolador(coef, x_points, y_points, polinomio_str):
    x_points = np.array(x_points, dtype=float)
    y_points = np.array(y_points, dtype=float)
    
    # Intervalo de plotagem com margem
    x_min, x_max = x_points.min() - 1, x_points.max() + 1
    x_range = np.linspace(x_min, x_max, 400)
    y_range = [newton_polynomial(x, x_points, coef) for x in x_range]
    
    # Criar DataFrames
    df_points = pd.DataFrame({"x": x_points, "y": y_points})
    df_poly = pd.DataFrame({"x": x_range, "y": y_range})
    
    # Criar figura
    fig = px.scatter(
        df_points,
        x="x",
        y="y",
        title="Polinômio Interpolador de Newton",
        labels={"x": "x", "y": "y"},
        template="plotly_dark",
        color_discrete_sequence=["cyan"]
    )
    
    # Adicionar curva do polinômio
    fig.add_trace(go.Scatter(
        x=df_poly["x"],
        y=df_poly["y"],
        mode="lines",
        line=dict(color="yellow", width=2),
        name="Polinômio Interpolador"
    ))
    
    # Adicionar equação
    fig.add_annotation(
        x=x_min + (x_max - x_min)*0.05,
        y=np.max(y_range),
        text=f"f(x) = {polinomio_str}",
        showarrow=False,
        font=dict(color="white", size=12),
        bgcolor="rgba(0,0,0,0.5)"
    )
    
    # Ajustar layout
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

#Lagrange

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

def polynomial(x, coef):
    result = 0
    degree = len(coef) - 1
    for i in range(len(coef)):
        result += coef[i] * x ** (degree - i)
    return result

def polinomio_interpolador(x_pontos, y_pontos, epsilon=1e-10):
    n = len(x_pontos)
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            A[i, j] = x_pontos[i] ** (n - j - 1)

    coef = np.linalg.solve(A, y_pontos)
    coef = np.array([0 if abs(c) < epsilon else c for c in coef])


    return coef


def plot_polinomio_interpolador(coef, x_pontos, y_pontos, epsilon=1e-10):

    # Converte os pontos para arrays NumPy (caso ainda não sejam)
    x_pontos = np.array(x_pontos, dtype=float)
    y_pontos = np.array(y_pontos, dtype=float)
    
    # Define a função polinomial utilizando np.polyval
    def polynomial(x):
        return np.polyval(coef, x)
    
    # Define o intervalo para plotar a curva (com margem de 1 unidade)
    x_min, x_max = x_pontos.min() - 1, x_pontos.max() + 1
    x_range = np.linspace(x_min, x_max, 400)
    y_range = polynomial(x_range)
    
    # Cria DataFrames para os pontos originais e para a curva do polinômio
    df_pontos = pd.DataFrame({"x": x_pontos, "y": y_pontos})
    df_polynomial = pd.DataFrame({"x": x_range, "y": y_range})
    
    # Cria o gráfico dos pontos originais
    fig = px.scatter(
        df_pontos, x="x", y="y",
        title="Polinômio Interpolador",
        labels={"x": "x", "y": "y"},
        template="plotly_dark",
        color_discrete_sequence=["cyan"]
    )
    
    # Adiciona a curva do polinômio
    fig.add_trace(go.Scatter(
        x=df_polynomial["x"],
        y=df_polynomial["y"],
        mode="lines",
        line=dict(color="yellow", width=2),
        name="Polinômio Interpolador"
    ))
    
    # Monta a string da equação do polinômio para anotação
    termos = []
    n = len(coef)
    for i, c in enumerate(coef):
        power = n - i - 1
        if abs(c) < epsilon:
            continue
        c_str = f"{round(c,4)}"
        if power == 0:
            termo = f"{c_str}"
        elif power == 1:
            termo = f"{c_str}x"
        else:
            termo = f"{c_str}x^{power}"
        termos.append(termo)
    equacao = " + ".join(termos)
    equacao = equacao.replace("+ -", "- ")
    
    # Adiciona anotação com a equação do polinômio
    fig.add_annotation(
        x=x_min + (x_max - x_min)*0.05,
        y=np.max(y_range),
        text=f"f(x) = {equacao}",
        showarrow=False,
        font=dict(color="white", size=12),
        bgcolor="rgba(0,0,0,0.5)"
    )
    
    # Atualiza o layout para melhorar a visualização
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig



def apresentando_interpolacao():
    graf = go.Figure()
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
        df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        

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
            coef, polinomio_str = polinomio_newton(x_pontos, y_pontos)
            graf = plot_newton_interpolador(coef, x_pontos, y_pontos, polinomio_str)
            st.write(f"Coeficientes do polinômio de Newton: {coef}")

            # Construir a string do polinômio
            polinomio_str = f"{coef[0]:.4f}"
            for i in range(1, len(coef)):
                termo = " * ".join([f"(x - {x_pontos[j]:.4f})" for j in range(i)])
                polinomio_str += f" + ({coef[i]:.4f}) * {termo}"

            st.write(f"Polinômio interpolador de Newton: {polinomio_str}")

            st.write("Método que encontra um polinômio que passa exatamente por todos os pontos dados, usando uma combinação de funções base. É eficiente, mas pode ser instável para grandes conjuntos de dados.")

            st.plotly_chart(graf)



    with col2:
        # Adicionando o método Lagrange
        with st.container(border=True):
            if st.button("Interpolador de Lagrange", use_container_width=True):
                if not df.empty:
                    st.session_state["Dados"] = df
                    st.session_state["lagrange"] = True
                    st.session_state["newton"] = False
                    st.empty()
                    st.rerun()

            st.write("Método incremental que usa diferenças divididas para construir o polinômio. Permite adicionar pontos sem recalcular os anteriores, sendo mais eficiente para novos dados.")

        # Adicionando o método Newton
        with st.container(border=True):
            if st.button("Interpolador de Newton", use_container_width=True):
                if not df.empty:
                    st.session_state["Dados"] = df
                    st.session_state["newton"] = True
                    st.session_state["lagrange"] = False
                    st.empty()
                    st.rerun()
            st.write("Método que encontra um polinômio que passa exatamente por todos os pontos dados, usando uma combinação de funções base. É eficiente, mas pode ser instável para grandes conjuntos de dados.")

    # Verificar se o método foi selecionado anteriormente
    

apresentando_interpolacao()
