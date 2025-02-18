import streamlit as st
from st_mathlive import mathfield
from PIL import Image
import re
import sympy as sp
import numpy as np
import plotly.graph_objects as go

def euler(func, x0, y0, xf, h):
    n = int((xf - x0) / h)
    xs = np.linspace(x0, xf, n+1)
    ys = np.zeros(n+1)
    ys[0] = y0

    for i in range(n):
        ys[i+1] = ys[i] + h * func(xs[i], ys[i])

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
    
    fig.show()

def processar_latex(latex):
    # Remove os primeiros 13 caracteres e o último
    expr = latex[13:-1]  # Remove os 13 primeiros e o último caractere
    # Substituir LaTeX para formato Python (se necessário)
    expr = expr.replace("^", "**")
    # Inserir * entre números e variáveis (ex: 3x -> 3*x)
    expr = re.sub(r'(?<=\d)(?=x)', '*', expr)
    # Inserir * entre variáveis e números (ex: x2 -> x*2)
    expr = re.sub(r'(?<=x)(?=\d)', '*', expr)
    return expr

im = Image.open("src/img/unifesp_icon.ico")

st.logo(
    im,
    link="https://portal.unifesp.br/",
    icon_image=im,
)

st.title("Derivação")

st.write("Insira a derivada no seguinte formato:")
st.latex(r"\frac{d}{dx}(x^2 + 3x + 2)")

# Usando mathfield para entrada de LaTeX
latex_input = mathfield(title="", value=r"\frac{d}{dx}(x^2 + 3x + 2)", mathml_preview=True)

# Botão para calcular
if st.button("Calcular"):
    if latex_input:  # Verificando se há entrada
        latex_input_str = latex_input[0]  # Acessando a string do LaTeX
        
        st.write(f"Equação inserida: {latex_input_str}")  # Exibir a entrada para depuração
        
        try:
            # Processar LaTeX para obter a função
            func_str = processar_latex(latex_input_str)
            
            if func_str:
                # Exibir a função processada
                st.write("Função processada:", func_str)
                # Usar sympy para transformar a string em uma expressão simbólica
                x, y = sp.symbols('x y')
                expressao = sp.sympify(func_str)

                # Gerar a função f(x, y) com lambdify
                f_func = sp.lambdify((x, y), expressao, 'numpy')

                # Visualizar a função
                st.write(f"Equação processada: {func_str}")
                st.latex(f"${func_str}$")

                # Exemplo de uso do método de Euler para gráficos
                x0, y0, xf, h = 0, 1, 10, 0.1  # Parâmetros do método de Euler
                plot_euler(f_func, x0, y0, xf, h)
            else:
                st.write("Não foi possível extrair coeficientes.")
        except Exception as e:
            st.write(f"Erro ao processar a equação: {e}")
    else:
        st.write("Por favor, insira um polinômio.")

# Adicionando CSS para personalizar o botão
st.markdown("""
    <style>
    .stButton>button {
        width: 300px;
        height: 50px;
        font-size: 18px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True)
