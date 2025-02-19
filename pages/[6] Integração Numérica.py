import streamlit as st
from st_mathlive import mathfield
import sympy as sp
import re
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.session_state['current_page'] = "Integração Numérica"

def corrigir_sintaxe(latex):
    if latex.startswith(r'\mathrm{') and latex.endswith('}'):
        latex = latex[len(r'\mathrm{'):-1]
    latex = latex.replace(" ", "").replace(",", "")
    latex = re.sub(r'\\int_(?!\{)(-?\d+)', r'\\int_{\1}', latex)
    latex = re.sub(r'\^(?!\{)(-?\d+)', r'^{\1}', latex)
    latex = re.sub(r'\)(dx)$', r') dx', latex)   
    return latex

def converter_fracao(expressao):
    return re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', expressao)

def limpar_latex(latex):
    from latex2sympy2 import latex2sympy
    return str(latex2sympy(latex))

def criar_função(latex):
    from sympy import lambdify, symbols
    latex_limpo = limpar_latex(latex)
    f = lambdify(symbols('x'), latex_limpo, modules=['numpy'])
    return f

latex_corrigido = corrigir_sintaxe(r'\int_{0}^{1}\left(x\right) dx')
padrão_integral = r'\\int_\{([^}]+)\}\^\{([^}]+)\}\\left\((.*?)\\right\)\s*dx'
match = re.search(padrão_integral, latex_corrigido)
lim_inferior, lim_superior, integrando = match.groups()
st.write(f"Limite inferior: {lim_inferior}")
st.write(f"Limite superior: {lim_superior}")
st.write(f"Integrando: {integrando}")


def extrair_integral(latex):
    try:
        latex_corrigido = corrigir_sintaxe(latex)
        
        # Padrão: \int_{a}^{b}\left( função \right) dx
        padrao_integral = r'\\int_\{([^}]+)\}\^\{([^}]+)\}\\left\((.*?)\\right\)\s*dx'
        match = re.search(padrao_integral, latex_corrigido)
        if not match:
            st.error("Formato incorreto da integral. Use: \\int_{a}^{b}\\left(f(x)\\right) dx")
            return None
        
        lim_inferior, lim_superior, integrando = match.groups()
        print(lim_inferior, lim_superior, integrando)

        st.write(f"Limite inferior: {lim_inferior}")
        st.write(f"Limite superior: {lim_superior}")
        st.write(f"Integrando: ${integrando}$")

        lim_inferior = criar_função(lim_inferior)
        lim_superior = criar_função(lim_superior)
        integrando = criar_função(integrando)
        return lim_inferior, lim_superior, integrando   
    except Exception as e:
        st.error(f"Erro ao processar a equação: {e}")
        return None
    
def trapezoidal_composta(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h * (y[0] + 2*np.sum(y[1:n]) + y[n]) / 2

def simpson_composta(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("O número de subintervalos (n) deve ser par para a regra de Simpson composta.")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    soma_impar = np.sum(y[1:n:2])   # índices 1, 3, 5, ..., n-1
    soma_par   = np.sum(y[2:n-1:2])   # índices 2, 4, 6, ..., n-2

    return h / 3 * (y[0] + y[n] + 4 * soma_impar + 2 * soma_par)

def plot_trapezoidal_composta_interactive(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)

    # Pontos densos para a curva real da função
    xx = np.linspace(a, b, 1000 + n)
    yy = f(xx)

    # Criação da figura com template dark
    fig = go.Figure()

    # Traço da função real (curva contínua)
    fig.add_trace(go.Scatter(x=xx, y=yy,
                             mode='lines',
                             line=dict(color='blue', width=2),
                             showlegend=False))

    # Traço dos pontos de aproximação e conexão entre eles
    fig.add_trace(go.Scatter(x=x, y=y,
                             mode='lines+markers',
                             marker=dict(color='red', size=8),
                             line=dict(color='red', width=1),
                             showlegend=False))

    # Desenhando os trapézios preenchidos
    for i in range(n):
        # Polígono do trapézio: vai do ponto (x[i], y[i]) a (x[i+1], y[i+1]) e volta para o eixo x (y=0)
        polygon_x = [x[i], x[i+1], x[i+1], x[i]]
        polygon_y = [y[i], y[i+1], 0, 0]
        fig.add_trace(go.Scatter(x=polygon_x, y=polygon_y,
                                 fill='toself',
                                 mode='none',
                                 fillcolor='rgba(255,165,0,0.4)',  # laranja com transparência
                                 line=dict(color='black'),
                                 showlegend=False))

    # Configuração final do layout
    fig.update_layout(template='plotly_dark',
                      title_text='Aproximação da Integral pela Regra dos Trapézios Composta',
                      xaxis_title='x',
                      yaxis_title='f(x)',
                      plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig)

def plot_simpson_composta_interactive(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("O número de subintervalos (n) deve ser par para a regra de Simpson composta.")

    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)

    # Pontos densos para a curva real da função
    xx = np.linspace(a, b, 1000)
    yy = f(xx)

    # Criação da figura com template dark
    fig = go.Figure()

    # Traço da função real (curva contínua)
    fig.add_trace(go.Scatter(x=xx, y=yy,
                             mode='lines',
                             line=dict(color='blue', width=2),
                             showlegend=False))

    # Traço dos pontos de aproximação e conexão entre eles
    fig.add_trace(go.Scatter(x=x, y=y,
                             mode='lines+markers',
                             marker=dict(color='red', size=8),
                             line=dict(color='red', width=1),
                             showlegend=False))

    # Aproximação por parábolas para cada par de subintervalos
    for i in range(0, n, 2):
        # Gerando pontos para a parábola entre x[i] e x[i+2]
        x_parabola = np.linspace(x[i], x[i+2], 100)
        # Ajuste quadrático pelos três pontos
        coef = np.polyfit([x[i], x[i+1], x[i+2]], [y[i], y[i+1], y[i+2]], 2)
        y_parabola = np.polyval(coef, x_parabola)

        # Criação do polígono para preencher a área entre a parábola e o eixo x
        # Concatena os pontos da curva com os pontos da linha base (y=0) em ordem reversa
        polygon_x = np.concatenate([x_parabola, x_parabola[::-1]]).tolist()
        polygon_y = np.concatenate([y_parabola, np.zeros_like(y_parabola)]).tolist()

        fig.add_trace(go.Scatter(x=polygon_x, y=polygon_y,
                                 fill='toself',
                                 mode='none',
                                 fillcolor='rgba(255,165,0,0.4)',  # laranja com transparência
                                 line=dict(color='black'),
                                 showlegend=False))

    # Configuração final do layout
    fig.update_layout(template='plotly_dark',
                      title_text='Aproximação da Integral pela Regra de Simpson Composta',
                      xaxis_title='x',
                      yaxis_title='f(x)',
                      plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig)


def principal():
    # --- Interface Streamlit ---
    st.title("Integração Numérica")
    st.write("Insira a integral no formato:")
    st.latex(r"\int_{a}^{b}\left(f(x)\right) dx")

    # Entrada via MathLive
    latex_input = mathfield(title="", value=r"\int_{0}^{1}\left(x\right) dx", mathml_preview=True)
    n = st.slider("Número de subdivisões", min_value=2, max_value=100, value=30, step=2)
    # Botão de calcular
    if st.button("Calcular", use_container_width=True):
        if latex_input:
            latex_str = latex_input[0]

            #st.write("Input original:", latex_str)
            resultado = extrair_integral(latex_str)
            if resultado:
                lim_inf, lim_sup, f = resultado
                r_trap = trapezoidal_composta(f, lim_inf(1), lim_sup(1), n)
                plot_trapezoidal_composta_interactive(f, lim_inf(1), lim_sup(1), n)
                r_trap = simpson_composta(f, lim_inf(1), lim_sup(1), n)
                plot_simpson_composta_interactive(f, lim_inf(1), lim_sup(1), n)
        else:
            st.write("Por favor, insira uma integral válida.")

principal()