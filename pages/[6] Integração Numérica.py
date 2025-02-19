import streamlit as st
from st_mathlive import mathfield
import sympy as sp
import re
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.session_state['current_page'] = "Integração Numérica"

def corrigir_sintaxe(latex):
    """
    Corrige a sintaxe do LaTeX para o formato esperado:
    - Remove \mathrm{...} se presente.
    - Remove espaços e vírgulas desnecessárias.
    - Garante que os limites inferior e superior estejam entre chaves.
    - Garante que haja um espaço antes do "dx".
    """
    # 1. Remover \mathrm{...} se houver
    if latex.startswith(r'\mathrm{') and latex.endswith('}'):
        latex = latex[len(r'\mathrm{'):-1]
    
    # 2. Remover espaços e vírgulas
    latex = latex.replace(" ", "").replace(",", "")
    
    # 3. Garantir que o limite inferior esteja entre chaves, se não estiver.
    latex = re.sub(r'\\int_(?!\{)(-?\d+)', r'\\int_{\1}', latex)
    
    # 4. Garantir que o limite superior esteja entre chaves, se não estiver.
    latex = re.sub(r'\^(?!\{)(-?\d+)', r'^{\1}', latex)
    
    # 5. Garantir que haja um espaço antes de "dx"
    latex = re.sub(r'\)(dx)$', r') dx', latex)
    
    return latex

def converter_fracao(expressao):
    """
    Converte frações LaTeX (\frac{a}{b}) para a sintaxe do SymPy ((a)/(b)).
    """
    return re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', expressao)

def extrair_integral(latex):
    try:
        latex_corrigido = corrigir_sintaxe(latex)
        st.write("LaTeX corrigido:", latex_corrigido)
        
        # Padrão: \int_{a}^{b}\left( função \right) dx
        padrao_integral = r'\\int_\{([^}]+)\}\^\{([^}]+)\}\\left\((.*?)\\right\)\s*dx'
        match = re.search(padrao_integral, latex_corrigido)
        if not match:
            st.error("Formato incorreto da integral. Use: \\int_{a}^{b}\\left(f(x)\\right) dx")
            return None
        
        lim_inferior, lim_superior, integrando = match.groups()
        
        # Processar o integrando:
        integrando = converter_fracao(integrando)  # Corrigir frações
        integrando = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', integrando)  # Adicionar "*"
        integrando = integrando.replace("^", "**")  # Converter expoentes
        integrando = re.sub(r'\*\*\{([^}]+)\}', r'**\1', integrando)  # Remover chaves de expoentes
        
        # Converter os limites para objetos do Sympy
        lim_inferior = sp.sympify(lim_inferior)
        lim_superior = sp.sympify(lim_superior)
        
        # Converter o integrando para uma expressão simbólica do Sympy
        x = sp.symbols('x')
        expressao_simbolica = sp.sympify(integrando, locals={'x': x})
        
        return lim_inferior, lim_superior, expressao_simbolica
    
    except Exception as e:
        st.error(f"Erro ao processar a equação: {e}")
        return None
    
def trapezoidal_composta(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h * (y[0] + 2*np.sum(y[1:n]) + y[n]) / 2

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









def principal():
    # --- Interface Streamlit ---
    st.title("Calculadora de Integrais")
    st.write("Insira a integral no formato:")
    st.latex(r"\int_{a}^{b}\left(f(x)\right) dx")

    # Entrada via MathLive
    latex_input = mathfield(title="", value=r"\int_{0}^{1}\left(x\right) dx", mathml_preview=True)
    n = st.slider("Número de subdivisões", min_value=2, max_value=100, value=30, step=2)
    # Botão de calcular
    if st.button("Calcular"):
        if latex_input:
            latex_str = latex_input[0]
            st.write("Input original:", latex_str)
            resultado = extrair_integral(latex_str)
            if resultado:
                lim_inf, lim_sup, func = resultado
                st.write(f"Limites: {lim_inf} a {lim_sup}")
                st.write(f"Função extraída: {func}")
                func = sp.lambdify('x', func, modules=['numpy'])
                lim_inf = float(lim_inf.evalf())
                lim_sup = float(lim_sup.evalf())
                r_trap = trapezoidal_composta(func, lim_inf, lim_sup, n)
                plot_trapezoidal_composta_interactive(func, lim_inf, lim_sup, n)
        else:
            st.write("Por favor, insira uma integral válida.")

principal()