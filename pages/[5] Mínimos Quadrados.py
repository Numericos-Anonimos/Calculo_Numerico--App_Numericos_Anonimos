import numpy as np
import pandas as pd
import plotly.express as px

import streamlit as st
st.session_state['current_page'] = "Mínimos Quadrados"
abas = ["Regressão Linear", "Modelo Exponencial", "Modelo Polinomial"]
abas = {i: j for i, j in zip(abas, st.tabs(abas))}

with abas["Regressão Linear"]:
    def baixar_arquivo(endereço):
        with open(endereço, "r", encoding="utf-8") as file:
            return file.read()
    arq = baixar_arquivo("resumos/[ 5 ] Mínimos Quadrados.md").split("<grafico>")


    st.write(arq[0])
    with st.container():
        def a (x, y):
            n, s1, s2, s3, s4 = len(x), sum(x*y), sum(x), sum(y), sum(x**2)
            return (n*s1 - s2*s3) / (n*s4 - s2**2)

        def b (x, y, a):
            return np.mean(y) - a*np.mean(x)

        def calcular_regressao_linear(x, a, b):
            return a * x + b

        def plotar_regressao_linear(x, y, a, b, xmin, xmax, n):
            x_line = np.linspace(xmin, xmax, n)
            y_line = calcular_regressao_linear(x_line, a, b)

            df_points = pd.DataFrame({'x': x, 'y': y})
            df_line = pd.DataFrame({'x': x_line, 'y': y_line})

            fig = px.scatter(df_points, x='x', y='y',
                            title='Regressão Linear',
                            labels={'x': 'Tempo(minutos)', 'y': 'Temperatura'},
                            template='plotly_dark',
                            color_discrete_sequence=['cyan'])

            fig.add_traces(px.line(df_line, x='x', y='y', color_discrete_sequence=['yellow']).data)

            return fig 

        x = np.array([1, 2, 3, 4, 5, 6, 7])
        y = np.array([13, 15, 20, 14, 15, 13, 10])
        a_ = a(x, y)
        b_ = b(x, y, a_)
        fig = plotar_regressao_linear(x, y, a_, b_, 0, 8, 100)

        st.write(fig)
        st.latex(f"f(x) = {a_:4f}x + {b_:.4f}")

    st.write("## Calculadora")
    df = st.data_editor(pd.DataFrame(columns=['x', 'y']), use_container_width=True, num_rows="dynamic")
    if st.button("Calcular"):
        x = df['x'].values.astype(float)
        y = df['y'].values.astype(float)
        a_ = a(x, y)
        b_ = b(x, y, a_)
        fig = plotar_regressao_linear(x, y, a_, b_, min(x), max(x), 100)
        st.write(fig)
        st.latex(f"f(x) = {a_:4f}x + {b_:.4f}")

with abas["Modelo Exponencial"]:
    st.write(arq[1])
    with st.container():
        def regressao_linear(x_vals, y_vals):
            n = len(x_vals)
            Sx = np.sum(x_vals)
            Sy = np.sum(y_vals)
            Sxx = np.sum(x_vals**2)
            Sxy = np.sum(x_vals * y_vals)
            b = (n * Sxy - Sx * Sy) / (n * Sxx - Sx**2)
            alpha = np.mean(y_vals) - b * np.mean(x_vals)

            return alpha, b

        def modelo_exponencial(x_val, a, b):
            return a * np.exp(b * x_val)

        def plotar_modelo_exponencial(x, y, a, b, xmin, xmax, n):
            x_line = np.linspace(xmin, xmax, n)
            y_line = modelo_exponencial(x_line, a, b)

            df_points = pd.DataFrame({'x': x, 'y': y})
            df_line = pd.DataFrame({'x': x_line, 'y': y_line})

            fig = px.scatter(df_points, x='x', y='y',
                            title='Modelo Exponencial',
                            labels={'x': 'x', 'y': 'y'},
                            template='plotly_dark',
                            color_discrete_sequence=['cyan'])

            fig.add_traces(px.line(df_line, x='x', y='y', color_discrete_sequence=['yellow']).data)

            return fig

        x = np.array([0.0, 1.5, 2.5, 3.5, 4.5])
        y = np.array([2.0, 3.6, 5.4, 8.1, 12.0])
        ln_y = np.log(y)

        alpha, b = regressao_linear(x, ln_y)
        a = np.exp(alpha)
        fig = plotar_modelo_exponencial(x, y, a, b, -1, 6, 100)
        st.write(fig)
        st.latex(f"f(x) = {a:.4f}e^{{{b:.4f}x}}")

    st.write("## Calculadora")
    df = st.data_editor(pd.DataFrame(columns=['x', 'y']), use_container_width=True, num_rows="dynamic", key="df_exponencial")
    if st.button("Calcular", key="btn_exponencial"):
        x = df['x'].values.astype(float)
        y = df['y'].values.astype(float)
        ln_y = np.log(y)

        alpha, b = regressao_linear(x, ln_y)
        a = np.exp(alpha)
        fig = plotar_modelo_exponencial(x, y, a, b, min(x), max(x), 100)
        st.write(fig)
        st.latex(f"f(x) = {a:.4f}e^{{{b:.4f}x}}")


with abas["Modelo Polinomial"]:
    st.write(arq[2])
    with st.container():
        def calcular_coeficientes(x, y, grau):
            n = len(x)
            A = np.zeros((n, grau + 1))
            for i in range(grau + 1):
                A[:, i] = x ** i

            AT_A = np.dot(A.T, A)
            AT_y = np.dot(A.T, y)
            coef = np.linalg.solve(AT_A, AT_y)
            return coef

        def calcular_polinomio(x, coef):
            y_pred = np.zeros_like(x)
            for i, c in enumerate(coef):
                y_pred += c * (x ** i)
            return y_pred

        def calcular_erro(x, y, coef):
            y_pred = calcular_polinomio(x, coef)
            erro = np.sum((y - y_pred) ** 2)
            return erro

        def plotar_multiplos_polinomios_interativo(x, y, coefs_labels, xmin, xmax, n):
            df_points = pd.DataFrame({'x': x, 'y': y})
            fig = px.scatter(df_points, x='x', y='y',
                            title='Comparação de Polinômios',
                            labels={'x': 'x', 'y': 'y'},
                            template='plotly_dark',
                            color_discrete_sequence=['cyan'])

            x_line = np.linspace(xmin, xmax, n)
            for coef, label in coefs_labels:
                y_line = calcular_polinomio(x_line, coef)
                fig.add_scatter(x=x_line, y=y_line, 
                                mode='lines', 
                                name=label,
                                line=dict(width=2))

            fig.update_layout(legend_title_text='Polinômios',
                            hovermode='x unified')

            return fig

        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([3.0, 15.0, 47.0, 99.0, 171.0, 263.0])
        coef_grau1 = calcular_coeficientes(x, y, 1)
        coef_grau2 = calcular_coeficientes(x, y, 2)
        fig = plotar_multiplos_polinomios_interativo(x, y, [(coef_grau1, 'Grau 1'), (coef_grau2, 'Grau 2')], 0, 5, 100)
        
        st.write(fig)
        st.latex(f"f(x) = {coef_grau1[0]:.4f}x + {coef_grau1[1]:.4f}")
        st.latex(f"f(x) = {coef_grau2[0]:.4f}x^2 + {coef_grau2[1]:.4f}x + {coef_grau2[2]:.4f}")

    st.write("## Calculadora")
    grau = st.multiselect("Grau do polinômio", [1, 2, 3, 4, 5], default=[2])    
    df = st.data_editor(pd.DataFrame(columns=['x', 'y']), use_container_width=True, num_rows="dynamic", key="df_polinomial")
    if st.button("Calcular", key="btn_polinomial"):
        x = df['x'].values.astype(float)
        y = df['y'].values.astype(float)
        coefs = []
        for g in grau:
            coefs.append((calcular_coeficientes(x, y, g), f"Grau {g}"))
        fig = plotar_multiplos_polinomios_interativo(x, y, coefs, min(x), max(x), 100)
        st.write(fig)
        for coef, label in coefs:
            st.latex(f"f(x) = {' + '.join([f'{c:.4f}x^{i}' for i, c in enumerate(coef)])}")
