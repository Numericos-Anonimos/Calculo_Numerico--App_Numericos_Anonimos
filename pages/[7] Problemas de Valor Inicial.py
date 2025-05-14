import streamlit as st
from st_mathlive import mathfield
from PIL import Image
import re
import sympy as sp
from sympy import lambdify, symbols, diff
from latex2sympy2 import latex2sympy
import numpy as np
import plotly.graph_objects as go


st.session_state['current_page'] = "Problemas de Valor Inicial"

st.markdown("""
    <style>
        a[href^="https://github.com"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)


def euler(func, x0, y0, xf, h):
    n = int((xf - x0) / h)
    xs = np.linspace(x0, xf, n+1)
    ys = np.zeros(n+1)
    ys[0] = y0

    for i in range(n):
        ys[i+1] = ys[i] + h * func(xs[i])

    return xs, ys

def plot_euler(func, x0, y0, xf, h):
    xs, ys = euler(func, x0, y0, xf, h)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode='lines+markers',
        name='Método de Euler',
        line=dict(color='lime', width=3),
        marker=dict(color='cyan', size=8)
    ))
    
    # Configurar layout com fundo preto e template "plotly_dark"
    fig.update_layout(
        title='Solução da EDO pelo Método de Euler',
        xaxis_title='x',
        yaxis_title='y',
        template='plotly_dark',
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig)




def processar_latex(latex):
    # Remove os primeiros 13 caracteres e o último
    expr = latex[13:-1]  # Remove os 13 primeiros e o último caractere
    return expr


def limpar_latex(latex):
    return str(latex2sympy(latex))


def preprocess_expr(expr):
    # Substitui ^ por ** para que o sympify entenda
    return expr.replace("^", "**")





def processar_latex(latex):
    # Remove os primeiros 13 caracteres e o último
    expr = latex[13:-1]  # Remove os 13 primeiros e o último caractere
    return expr

st.title("Problemas de Valor Inicial")

st.write("Insira a derivada no seguinte formato (incluindo os parênteses no início e no fim):")
st.latex(r"\frac{d}{dx}(x^2 + 3x + 2)")

# Usando mathfield para entrada de LaTeX
latex_input = mathfield(title="", value=r"\frac{d}{dx}(x^2 + 3x + 2)", mathml_preview=True)


gx0 = st.slider("x0 (Ponto inicial)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
gy0 = st.slider("y0 (Valor inicial de y)", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
gxf = st.slider("xf (Ponto final)", min_value=1.0, max_value=20.0, value=10.0, step=0.1)
gh = st.slider("h (Tamanho do passo)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

def quebrar_derivada(latex):
    latex = latex.split(r'{dx}')[1]
    return limpar_latex(latex)

def limpar_latex(latex):
    from latex2sympy2 import latex2sympy
    return str(latex2sympy(latex))

def criar_função(latex):
    from sympy import lambdify, symbols
    f = lambdify(symbols('x'), latex, modules=['numpy'])
    return f

# Botão para calcular
if st.button("Calcular", use_container_width=True):
    if latex_input:  # Verificando se há entrada
        latex_input_str = latex_input[0]  # Acessando a string do LaTeX       
        #st.write(f"Equação inserida: {latex_input_str}")  # Exibir a entrada para depuração
        try:
            func_str = quebrar_derivada(latex_input_str)
            st.write(f"Função extraída: {func_str}")
            if func_str:
                f = criar_função(func_str)
                plot_euler(f, gx0, gy0, gxf, gh)
            else:
                st.write("Não foi possível extrair coeficientes.")
        except Exception as e:
            st.write(f"Erro ao processar a equação: {e}")
    else:
        st.write("Por favor, insira uma equação válida.")

