import streamlit as st
from st_mathlive import mathfield
import sympy as sp
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def criar_dataframe(iterations, metodo):
    data = []
    for k, x in enumerate(iterations):
        for i, xi in enumerate(x):
            data.append({
                "Iteração": k, "Variável": f"$x_{{{i+1}}}$",
                "Valor": xi, "Método": metodo
            })
    return pd.DataFrame(data)

def jacobi_solver(A, b, x0, tol=1e-6, max_iter=100):
    n = len(b)
    x = x0.copy()
    iterations = [x.copy()]
    for k in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
        iterations.append(x_new.copy())
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
    return x, iterations

def plot_jacobi_gauss_seidel(jacobi=None, gauss_seidel=None):
    nome = "Comparação entre Jacobi e Gauss-Seidel" if jacobi and gauss_seidel else ("Método de Jacobi" if jacobi else "Método de Gauss-Seidel")
    fig = px.line(title=nome)
    
    if jacobi:
        df_jacobi = criar_dataframe(jacobi, "Jacobi")
        fig.add_traces(px.line(df_jacobi, x="Iteração", y="Valor",
                               color="Variável",
                               markers=True).data)
    if gauss_seidel:
        df_gauss_seidel = criar_dataframe(gauss_seidel, "Gauss-Seidel")
        fig.add_traces(px.line(df_gauss_seidel, x="Iteração", y="Valor",
                               color="Variável",
                               markers=True, line_dash_sequence=['dash']).data)
    
    return fig





def principal():
    # --- Interface Streamlit ---
    st.title("Sistemas Lineares")
    st.write("Insira a matriz das variáveis e incógnitas seguindo o formato:")
    st.latex(r"\mathrm{\begin{pmatrix}A11 & A12 & A13 & B1\\ A21 & A22 & A23 & B2\\ A31 & A32 & A33 & B3\end{pmatrix}}")

    # Entrada via MathLive
    latex_input = mathfield(title="", value=r"\mathrm{\begin{pmatrix}1 & 2 & 3 & 4\\ 1 & 2 & 3 & 4\\1 & 2 & 3 & 4\end{pmatrix}}", mathml_preview=True)
    # Botão de calcular
    if st.button("Calcular"):
        if latex_input:
            latex_str = latex_input[0]
            st.write("Input original:", latex_str)
            # Remover comandos LaTeX e caracteres extras
            data_str = re.sub(r"\\(mathrm|begin|end){.*?}", "", latex_str)  # Remove comandos LaTeX
            data_str = re.sub(r"[{}]", "", data_str)  # Remove chaves '{}'
            data_str = data_str.strip()  # Remove espaços extras no início e fim

            # Separar as linhas e processar os valores
            rows = [list(map(int, row.strip().split("&"))) for row in data_str.split("\\\\")]

            # Converter para numpy arrays
            matriz = np.array([row[:-1] for row in rows])  # Matriz sem a última coluna
            vetor = np.array([row[-1] for row in rows])  # Última coluna

            # Exibir resultados
            x0 = [0 for i in vetor]
            st.write(matriz)
            st.write(vetor)
            st.write(x0)
            x_jacobi, iter_jacobi = jacobi_solver(matriz, vetor, x0, tol=1e-6, max_iter=100)
            fig = plot_jacobi_gauss_seidel(iter_jacobi)
            st.plotly_chart(fig)

        else:
            st.write("Por favor, insira uma integral válida.")

principal()