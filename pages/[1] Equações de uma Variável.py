import streamlit as st
from st_mathlive import mathfield
import sympy as sp
import re
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.session_state['current_page'] = "Equações de uma Variável"

st.html('''
<style>
    #MainMenu {visibility: collapsed;}
    footer {visibility: hidden;}
    header {visibility: hidden;} 
</style>''')


# Função para limpar o LaTeX
def limpar_latex(latex):
    from latex2sympy2 import latex2sympy
    import sympy as sp
    return str(latex2sympy(latex))

def criar_função(latex):
    from sympy import lambdify, symbols
    latex_limpo = limpar_latex(latex)
    f = lambdify(symbols('x'), latex_limpo, modules=['numpy'])
    return f

def bisseccao(f, ini, fim, max_iter):
    if f(ini) * f(fim) >= 0:raise ValueError("Não há raiz no intervalo [a, b].")
    pontos = []
    while abs(fim - ini) > 1e-9:
        meio = (ini + fim) / 2.0
        pontos.append(meio)
        if f(ini) * f(meio) < 0: fim = meio
        else: ini = meio
        if len(pontos) > max_iter: break
    return meio, np.array(pontos)

def plotar_bisseccao(f, ini, fim, raiz, pontos, xmin, xmax, n):
    x = np.linspace(xmin, xmax, n)
    y = f(x)
    fig = px.line(x = x, y = y, title = 'Método da Bisseção',
                  labels = {'x': 'x', 'y': 'f(x)'},
                  template='plotly_dark')
    fig.add_scatter(x = pontos, y = [f(p) for p in pontos],
        mode = 'markers+lines', marker = dict(color='cyan', size=8),
        line = dict(color='gray', dash='dot'), name = 'Passos da Bissecção')
    fig.add_scatter(x = [raiz], y = [0], mode = 'markers',
        marker = dict(color='red', size=12), name = f'Raiz Final: {raiz:.9f}')
    fig.add_vline(x=ini, line_dash="dash", line_color="green", annotation_text="Início (a)")
    fig.add_vline(x=fim, line_dash="dash", line_color="green", annotation_text="Fim (b)")
    return fig

def secantes(f, x0, x1, tol=1e-6, max_iter=100):
    pontos = [x0, x1]
    while abs(f(x1)) > tol or max_iter > 0:
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        pontos.append(x2)
        x0, x1 = x1, x2
        max_iter -= 1
        if abs(f(x1) - f(x0)) < tol: break
    return x2, np.array(pontos)

def plot_secantes(f, raiz, pontos, xmin, xmax, n=100):
    x = np.linspace(xmin, xmax, n)
    y = f(x)
    df_func = pd.DataFrame({'x': x, 'f(x)': y,})
    df_pontos = pd.DataFrame({'x': pontos, 'y': f(pontos), 'Iterações': range(len(pontos))})
    fig = px.line(df_func, x='x', y='f(x)', title='Método das Secantes', 
                  labels={'x': 'x', 'f(x)': 'f(x)'}, template='plotly_dark')
    fig.add_scatter(x=df_pontos['x'], y=df_pontos['y'], mode='lines', 
                    marker=dict(color='cyan', size=8), name='Iterações')
    fig.add_scatter(x=[pontos[0]], y=[f(pontos[0])], mode='markers', 
                    marker=dict(color='orange', size=12, symbol='circle'),
                    name='Chute inicial (x0)')
    fig.add_scatter(x=[pontos[1]], y=[f(pontos[1])], mode='markers', 
                    marker=dict(color='orange', size=12, symbol='diamond'),
                    name='Chute inicial (x1)')
    fig.add_scatter(x=[raiz], y=[f(raiz)], mode='markers',
                    marker=dict(color='yellow', size=12, symbol='star-diamond'),
                    name='Raiz Aproximada')
    return fig

def derivada(latex):
    from sympy import lambdify, symbols, diff
    latex_limpo = limpar_latex(latex)
    derivada = diff(latex_limpo, symbols('x'))
    return lambdify(symbols('x'), derivada, modules=['numpy'])

def heron(f, df, x0, tol=1e-6, max_iter=100):
    x = x0
    pontos = [x]
    for _ in range(max_iter):
        x = x - f(x)/df(x)
        pontos.append(x)
        if abs(pontos[-1] - pontos[-2]) < tol:
            return np.array(x), np.array(pontos)
    return np.array(x), np.array(pontos)

def plot_heron(f, raiz, pontos, xmin, xmax, n=1000):
    x = np.linspace(xmin, xmax, n)
    y = f(x)
    df_func = pd.DataFrame({'x': x, 'y': y, 'trace': 'f(x)'})
    df_zero = pd.DataFrame({'x': x, 'y': np.zeros_like(x), 'trace': 'y = 0'})
    df_pontos = pd.DataFrame({'x': pontos, 'y': f(pontos), 'trace': 'Iterações'})
    df_lines = pd.concat([df_func, df_zero])
    fig = px.line(df_lines, x='x', y='y', color='trace',
                  title='Método de Heron (Newton)',
                  labels={'x': 'x', 'y': 'f(x)'},
                  template='plotly_dark')
    fig.add_scatter(x=df_pontos['x'], y=df_pontos['y'],
                    mode='markers', name='Iterações',
                    marker=dict(color='cyan'))
    fig.add_scatter(x=[raiz], y=[f(raiz)], mode='markers', 
                    marker=dict(color='yellow', size=12, symbol='diamond'),
                    name='Ponto Final')   
    return fig



def principal():
    graf = go.Figure()
    
    if "calcular" in st.session_state and st.session_state["calcular"]:
        if "f" in st.session_state and st.session_state["f"]:
            st.header("Métodos de aproximação de raízes:")
            st.latex(st.session_state["latex"])
            col1, col2, col3 = st.columns(3)

            # Método da Bissecção
            if "metodo_bisseccao" in st.session_state and st.session_state["metodo_bisseccao"]:
                st.markdown("---")
                st.write("### Método escolhido: Bissecção")

                ini = st.number_input("Insira o limite inferior do intervalo", value=0.0)
                fim = st.number_input("Insira o limite superior do intervalo", value=1.0)
                max_iter = st.number_input("Insira o número máximo de iterações", value=100, step=1)

                col_aplicar, col_voltar = st.columns([0.2, 0.2])

                with col_aplicar:
                    if st.button("Aplicar", use_container_width=True):
                        f = st.session_state["f"]
                        try:
                            raiz, pontos = bisseccao(f, ini, fim, max_iter)
                            graf = plotar_bisseccao(f, ini, fim, raiz, pontos, ini-1, fim+1, 1000)
                            st.session_state["raiz"] = raiz
                            st.session_state["pontos"] = pontos
                            st.session_state["encontrou_resultado"] = True
                            st.session_state["grafico"] = graf
                            st.rerun()
                        except ValueError as e:
                            st.write(f"Erro: {e}")
                            st.session_state["encontrou_resultado"] = False

                with col_voltar:
                    if st.button("Voltar", key="btn_voltar", use_container_width=True):
                        st.session_state["metodo_bisseccao"] = False
                        st.session_state["metodo_newton"] = False
                        st.session_state["metodo_secante"] = False
                        st.session_state["calcular"] = False
                        st.session_state["encontrou_resultado"] = False
                        st.empty()
                        st.rerun()

            # Método de Newton
            elif "metodo_newton" in st.session_state and st.session_state["metodo_newton"]:
                st.markdown("---")
                st.write("### Método escolhido: Newton")

                x0 = st.number_input("Insira o valor inicial 'x0'", value=0.0)
                n = st.number_input("Insira o número máximo de iterações 'n'", value=10)

                col4, col5 = st.columns([1, 1])  # Criando colunas para alinhar os botões

                with col4:
                    if st.button("Aplicar", use_container_width=True):
                        f = st.session_state["f"]
                        df = st.session_state["df"]
                        try:
                            raiz, pontos = heron(f, df, x0)
                            graf = plot_heron(f, raiz, pontos, x0-2, x0+2, n)
                            st.session_state["raiz"] = raiz
                            st.session_state["pontos"] = pontos
                            st.session_state["encontrou_resultado"] = True
                            st.session_state["grafico"] = graf
                            st.rerun()
                        except Exception as e:
                            st.write(f"Erro: {e}")

                with col5:
                    if st.button("Voltar", use_container_width=True):
                        st.session_state["metodo_bisseccao"] = False
                        st.session_state["metodo_newton"] = False
                        st.session_state["metodo_secante"] = False
                        st.session_state["calcular"] = False
                        st.session_state["encontrou_resultado"] = False
                        st.empty()
                        st.rerun()  # Rerun para voltar ao estado inicial

            # Método da Secante
            elif "metodo_secante" in st.session_state and st.session_state["metodo_secante"]:
                st.markdown("---")
                st.write("### Método escolhido: Secante")

                # Entrada dos valores para x0, x1, e número de iterações
                x0 = st.number_input("Insira o valor inicial 'x0'", value=0.0)
                x1 = st.number_input("Insira o valor inicial 'x1'", value=1.0)
                n = st.number_input("Insira o número máximo de iterações 'n'", value=10)
                f = st.session_state["f"]

                col4, col5 = st.columns([1, 1])  # Criando colunas para alinhar os botões

                with col4:
                    if st.button("Aplicar", use_container_width=True):
                        try:
                            raiz, pontos = secantes(f, x0, x1)
                            graf = plot_secantes(f, raiz, pontos, x0-2, x1+2)
                            st.session_state["raiz"] = raiz
                            st.session_state["pontos"] = pontos
                            st.session_state["encontrou_resultado"] = True
                            st.session_state["grafico"] = graf
                            st.rerun()
                        except ValueError as e:
                            st.write(f"Erro: {e}")

                with col5:
                    if st.button("Voltar", use_container_width=True):
                        st.session_state["metodo_bisseccao"] = False
                        st.session_state["metodo_newton"] = False
                        st.session_state["metodo_secante"] = False
                        st.session_state["calcular"] = False
                        st.session_state["encontrou_resultado"] = False
                        st.empty()
                        st.rerun()

            if "metodo_bisseccao" in st.session_state and st.session_state["metodo_bisseccao"] and "encontrou_resultado" in st.session_state and st.session_state["encontrou_resultado"]:
                st.markdown("---")
                st.write("### Resolvendo por: Bissecção")
                if "raiz" in st.session_state:
                    st.write(f"Raiz encontrada: {st.session_state['raiz']}")
                    st.write(f"Pontos intermediários: {st.session_state['pontos']}")
                    st.plotly_chart(st.session_state['grafico'])

            if "metodo_newton" in st.session_state and st.session_state["metodo_newton"] and "encontrou_resultado" in st.session_state and st.session_state["encontrou_resultado"]:
                st.markdown("---")
                st.write("### Resolvendo por: newton")
                if "raiz" in st.session_state:
                    st.write(f"Raiz encontrada: {st.session_state['raiz']}")
                    st.write(f"Pontos intermediários: {st.session_state['pontos']}")
                    st.plotly_chart(st.session_state['grafico'])

            if "metodo_secante" in st.session_state and st.session_state["metodo_secante"] and "encontrou_resultado" in st.session_state and st.session_state["encontrou_resultado"]:
                st.markdown("---")
                st.write("### Resolvendo por: secante")
                if "raiz" in st.session_state:
                    st.write(f"Raiz encontrada: {st.session_state['raiz']}")
                    st.write(f"Pontos intermediários: {st.session_state['pontos']}")
                    st.plotly_chart(st.session_state['grafico'])

            # Coluna para os botões de escolha de métodos
            with col1:
                if st.button("**Método da Bissecção**", use_container_width=True):
                    st.session_state["metodo_bisseccao"] = True
                    st.session_state["metodo_newton"] = False
                    st.session_state["metodo_secante"] = False
                    st.session_state["encontrou_resultado"] = False
                    st.empty()
                    st.rerun()

            with col2:
                if st.button("**Método de Newton**", use_container_width=True):
                    st.session_state["metodo_newton"] = True
                    st.session_state["metodo_bisseccao"] = False
                    st.session_state["metodo_secante"] = False
                    st.session_state["encontrou_resultado"] = False
                    st.empty()
                    st.rerun()

            with col3:
                if st.button("**Método da Secante**", use_container_width=True):
                    st.session_state["metodo_secante"] = True
                    st.session_state["metodo_bisseccao"] = False
                    st.session_state["metodo_newton"] = False
                    st.session_state["encontrou_resultado"] = False
                    st.empty()
                    st.rerun()

    else:
        st.title("Aproximação de Polinômios")
        st.write("Insira a equação no seguinte formato:")
        st.latex(r"x^2 + x - 1")
        latex_input = mathfield(title="", value=r"x^2 + x - 1", mathml_preview=True)

        if st.button("Calcular", use_container_width=True):
            if latex_input:
                latex_input_str = latex_input[0]
                match = re.match(r'\\mathrm\{(.+)\}', latex_input_str)
                if match:
                    latex_input_str = match.group(1)
                latex_input_str = latex_input_str.replace(r"\mathrm{", "").rstrip("}")

                try:
                    print(fr'{latex_input_str}')
                    f = criar_função(latex_input_str)
                    st.write(f"Função inserida: {f}")
                    df = derivada(latex_input_str)
                    st.write(f"Derivada da função: {df}")
                    st.session_state["f"] = f
                    st.session_state["df"] = df
                    st.session_state["latex"] = latex_input_str
                    st.session_state["calcular"] = True
                    st.rerun()
                except Exception as e:
                    st.write(f"Erro ao processar a equação: {e}")
            else:
                st.write("Por favor, insira um polinômio.")



principal()
