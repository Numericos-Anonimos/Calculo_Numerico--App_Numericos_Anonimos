import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Leia o arquivo de texto A.md
def read_file():
    with open('A.md', 'r') as file:
        data = file.read()
    st.write(data)

read_file()

def funcao(x):
    return x**2 - 2

def derivada(x):
    return 2*x

def heron(x0, tol=1e-6):
    x = x0
    pontos = []
    while abs(funcao(x)) > tol:
        x = x - funcao(x)/derivada(x)
        pontos.append(x)
    return x, pontos

raiz, iteracoes = heron(1)

x_vals = np.linspace(0, 2, 400)
y_vals = funcao(x_vals)
df_funcao = pd.DataFrame({
    "x": x_vals,
    "y": y_vals,
    "trace": "f(x) = x² - 2"
})

df_zero = pd.DataFrame({
    "x": x_vals,
    "y": np.zeros_like(x_vals),
    "trace": "y = 0"
})

df_iter = pd.DataFrame({
    "x": iteracoes,
    "y": [funcao(x) for x in iteracoes],
    "trace": "Iterações"
})

df_lines = pd.concat([df_funcao, df_zero])

fig = px.line(df_lines, x="x", y="y", color="trace",
              title="Método de Heron (Newton) para f(x) = x² - 2",
              labels={"x": "x", "y": "f(x)"},
              template="plotly_dark")

fig_iter = px.scatter(df_iter, x="x", y="y", color="trace", template="plotly_dark",
                        color_discrete_map={"Iterações": "cyan"})
for trace in fig_iter.data:
    fig.add_trace(trace)

fig.add_scatter(x=[raiz], y=[funcao(raiz)], mode="markers",
                marker=dict(color="yellow", size=12, symbol="diamond"),
                name="Ponto Final")

st.plotly_chart(fig)