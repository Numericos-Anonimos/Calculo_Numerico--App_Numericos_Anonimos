import pandas as pd
from datetime import datetime
import streamlit as st
from PIL import Image
import plotly.express as px
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from metodos import *

st.session_state['current_page'] = "Home"

# Verificar se a chave da página "Bisseção" foi marcada na sessão
if "Bissecao" in  st.session_state and st.session_state["Bissecao"]:
    def read_file():
        with open('resumos/[ 1 ] Bisseção.md', 'r', encoding="utf-8") as file:
            data = file.read()
        st.markdown(data)  # Exibe com formatação correta

    read_file()

    
    f = lambda x: x**2 - 2

    # Executando o método da bisseção
    raiz, pontos = bisseccao(f, 1.0, 2.0)
    # Gerando o gráfico
    fig = plotar_bisseccao(f, 1.0, 2.0, raiz, pontos, 0.0, 3.0, 1000)

    # Exibindo no Streamlit
    st.title(f"Visualização do Método da Bisseção:  ")
    st.write(f"Função plotada: {format_function_markdown(f)}")
    st.plotly_chart(fig, use_container_width=True)
    

    if st.button("Voltar"):
        st.session_state["Bissecao"] = False
        st.empty()
        st.rerun()

elif "Newton" in  st.session_state and st.session_state["Newton"]:
    def read_file():
        with open('resumos/[ 1 ] Heron Newton.md', 'r', encoding="utf-8") as file:
            data = file.read()
        st.write(data)

    read_file()
    st.markdown("---")
    f = lambda x: x**2 - 2
    df = lambda x: 2*x
    x0 = 1

    # Executando o método de Heron (Newton)
    raiz, pontos = heron(f, df, x0)

    # Definindo o intervalo
    xmin, xmax = 0, 2

    # Gerando o gráfico
    fig = plot_heron(f, raiz, pontos, xmin, xmax)

    # Exibindo o gráfico no Streamlit
    st.title("Visualização do Método de Heron (Newton) ")
    st.write(f"Função plotada:  {format_function_markdown(f)}")
    st.plotly_chart(fig, use_container_width=True)



    if st.button("Voltar"):
        st.session_state["Newton"] = False
        st.empty()
        st.rerun()
elif "Secante" in  st.session_state and st.session_state["Secante"]:
    def read_file():
        with open('resumos/[ 1 ] Método das Secantes.md', 'r', encoding="utf-8") as file:
            data = file.read()
        st.write(data)

    read_file()
    x0, x1 = 1, 2
    f = lambda x: x**2 - 2

    # Executando o método das secantes
    raiz, pontos = secantes(f, x0, x1)

    # Definindo o intervalo
    xmin, xmax = 0, 3

    # Gerando o gráfico
    fig = plot_secantes(f, raiz, pontos, xmin, xmax)

    # Exibindo o gráfico no Streamlit
    st.title("Visualização do Método das Secantes ")
    st.write(f"Função plotada: {format_function_markdown(f)}")
    st.plotly_chart(fig, use_container_width=True)


    if st.button("Voltar"):
        st.session_state["Secante"] = False
        st.empty()
        st.rerun()
elif "sistemas_lineares" in  st.session_state and st.session_state["sistemas_lineares"]:
    def read_file():
        with open('resumos/[ 2 ] Jacobi e Gauss-Seidel.md', 'r', encoding="utf-8") as file:
            data = file.read()
        st.write(data)

    read_file()
    st.markdown("---")

    A = np.array([[4.0, 2.0],
              [1.0, 3.0]])
    b = np.array([9.0, 8.0])
    x0 = np.array([1.0, 1.0])

    # Executando o método de Jacobi
    x_jacobi, iter_jacobi = jacobi_solver(A, b, x0, tol=1e-6, max_iter=25)

    # Gerando o gráfico
    fig = plot_jacobi(iter_jacobi)

    # Exibindo o gráfico no Streamlit
    st.title("Visualização do Método de Jacobi")
    st.write(f"A = {A} / b = {b} / x0 = {x0}")
    st.plotly_chart(fig, use_container_width=True)




    x_gauss_seidel, iter_gauss_seidel = gauss_seidel_solver(A, b, x0, tol=1e-6, max_iter=25)

    # Gerando o gráfico para Gauss-Seidel
    df_gauss_seidel = criar_dataframe(iter_gauss_seidel, "Gauss-Seidel")
    fig_gauss_seidel = px.line(df_gauss_seidel, x="Iteração", y="Valor", color="Variável", markers=True, title="", template="plotly_dark")

    # Exibindo o gráfico no Streamlit para Gauss-Seidel
    st.title("Visualização do Método de Gauss-Seidel")
    st.write(f"A = {A} / b = {b} / x0 = {x0}")
    st.plotly_chart(fig_gauss_seidel, use_container_width=True)

    if st.button("Voltar"):
        st.session_state["sistemas_lineares"] = False
        st.empty()
        st.rerun()

    
elif "interpol" in  st.session_state and st.session_state["interpol"] :
    def read_file():
        with open('resumos/[ 4 ] Interpolação.md', 'r', encoding="utf-8") as file:
            data = file.read()
        st.write(data)

    read_file()
    st.markdown("---")
    x = np.array([1, 2, 3, 4], dtype=float)
    y = np.array([2, 3, 5, 4], dtype=float)

    # Calculando os coeficientes do polinômio interpolador
    coef = lagrange(x, y)

    # Gerando o gráfico
    fig = plotar_polinomio(x, y, coef, 0, 5, 100)

    # Exibindo o gráfico no Streamlit
    st.title("Interpolação de Lagrange")
    st.write(f"Pontos de entrada: x = {x}, y = {y}")
    st.plotly_chart(fig, use_container_width=True)

    # Adicionar o método de Newton

    if st.button("Voltar"):
        st.session_state["interpol"] = False
        st.empty()
        st.rerun()
elif "minimos" in  st.session_state and st.session_state["minimos"] :
    def read_file():
        with open('resumos/[ 6 ] Integração.md', 'r', encoding="utf-8") as file:
            data = file.read()
        st.write(data)

    read_file()
    st.markdown("---")

    x = np.array([1, 2, 3, 4, 5, 6, 7])
    y = np.array([13, 15, 20, 14, 15, 13, 10])

    # Cálculo dos coeficientes
    a_, b_ = calcular_coeficientes(x, y)

    # Gerando o gráfico
    fig = plotar_regressao_linear(x, y, a_, b_, 0, 8, 100)

    # Exibindo no Streamlit
    st.title("Regressão Linear Simples")
    st.write(f"Coeficiente angular (a): {a_:.4f}  /  Coeficiente linear (b): {b_:.4f}  /  Equação da reta: y = {a_:.4f}x + {b_:.4f}")
    st.plotly_chart(fig, use_container_width=True)

    x = np.array([0.0, 1.5, 2.5, 3.5, 4.5])
    y = np.array([2.0, 3.6, 5.4, 8.1, 12.0])
    ln_y = np.log(y)

    # Cálculo dos coeficientes
    alpha, b = regressao_linear(x, ln_y)
    a = np.exp(alpha)

    # Gerando o gráfico
    fig = plotar_modelo_exponencial(x, y, a, b, -1, 6, 100)

    # Exibindo no Streamlit
    st.title("Ajuste de Modelo Exponencial")
    st.write(f"Coeficiente 'a': {a:.4f}  /  Coeficiente 'b': {b:.4f}  /  Equação do modelo: y = {a:.4f} * exp({b:.4f} * x)")
    st.plotly_chart(fig, use_container_width=True)

    st.title("Ajuste de Polinômios via Mínimos Quadrados")

    # Definição dos dados
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([3.0, 15.0, 47.0, 99.0, 171.0, 263.0])

    # Seleção interativa do grau do polinômio
    grau_max = st.slider("Escolha o grau do polinômio:", min_value=1, max_value=5, value=2)

    # Cálculo dos coeficientes para os graus escolhidos
    coefs_labels = [(calcular_coeficientes2(x, y, i), f'Grau {i}') for i in range(1, grau_max + 1)]

    # Gerando o gráfico
    fig = plotar_multiplos_polinomios_interativo(x, y, coefs_labels, 0, 5, 100)

    # Exibição no Streamlit
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Voltar"):
        st.session_state["minimos"] = False
        st.empty()
        st.rerun()
elif "integra" in  st.session_state and st.session_state["integra"] :
    def read_file():
        with open('resumos/[ 6 ] Integração.md', 'r', encoding="utf-8") as file:
            data = file.read()
        st.write(data)

    read_file()
    st.markdown("---")
    st.title("Visualização da Regra dos Trapézios Composta")

    f = lambda x:  1/(4 + np.sin(20*x))
    a = st.slider("Valor inicial (a)", 0.0, 5.0, 0.0)
    b = st.slider("Valor final (b)", 0.0, 5.0, 3.0)
    n = st.slider("Número de subdivisões (n)", 1, 100, 30)

    fig = plot_trapezoidal_composta_interactive(f, a, b, n)

    st.plotly_chart(fig)
    
    st.title("Visualização da Regra de Simpson Composta")

    a = st.slider("Limite inferior (a)", 0.0, 10.0, 0.0)
    b = st.slider("Limite superior (b)", 0.0, 10.0, 3.0)
    n = st.slider("Número de subintervalos (n, deve ser par)", 2, 100, 30, step=2)

    f = lambda x: 1 / (4 + np.sin(20*x))

    resultado = simpson_composta(f, a, b, n)
    st.write(f"Valor aproximado da integral: {resultado:.6f}")

    fig = plot_simpson_composta_interactive(f, a, b, n)
    st.plotly_chart(fig)



    
    if st.button("Voltar"):
        st.session_state["integra"] = False
        st.empty()
        st.rerun()
elif "edo1" in  st.session_state and st.session_state["edo1"] :
    def read_file():
        with open('resumos/[ 7 ] Método de Euler.md', 'r', encoding="utf-8") as file:
            data = file.read()
        st.write(data)

    read_file()
    st.markdown("---")

    x0 = 0.0
    y0 = 1.0
    xf = 2.0
    h = 0.1

    st.title("Visualização do Método de Euler")
    fig = plot_euler(f, x0, y0, xf, h)
    st.write(f"x0: {x0} / y0: {y0} / xf: {xf} / h: {h}")
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Voltar"):
        st.session_state["edo1"] = False
        st.empty()
        st.rerun()
elif "edo2" in  st.session_state and st.session_state["edo2"] :
    def read_file():
        with open('resumos/[ 7 ] Runge Kutta.md', 'r', encoding="utf-8") as file:
            data = file.read()
        st.write(data)

    read_file()
    st.markdown("---")

    t0 = 0.0
    t_end = 2.0
    y0 = 1.0
    dt = 0.1

    st.title("Visualização do Método de Runge Kutta")
    fig = plot_runge_kutta(f, y0, t0, t_end, dt)
    st.write(f"t0: {t0} / t_end: {t_end} / y0: {y0} / dt: {dt}")
    st.plotly_chart(fig, use_container_width=True)
    

    if st.button("Voltar"):
        st.session_state["edo2"] = False
        st.empty()
        st.rerun()
else:
    st.title("Numéricos Anônimos")
    # Exibir os botões se a página "Bisseção" não foi escolhida
    col1, col2 = st.columns(2, gap='small')

    with col1:
        with st.container(border=True):
            st.page_link("pages/[1] Equações de uma Variável.py", label="🔢 **Equações de Uma Variável**")
            st.caption("Aproximação de polinômios")
        with st.container(border=True):
            st.page_link("pages/[2] Sistemas Lineares.py", label="📐 **Sistemas Lineares**")
            st.caption("Resolução de Sistemas Lineares com Matrizes")
        with st.container(border=True):
            st.page_link("pages/[4] Interpolação.py", label="📊 **Interpolação**")
            st.caption("Construção de Polinômio Interpolador")

    with col2:
        with st.container(border=True):
            st.page_link("pages/[7] Problemas de Valor Inicial.py", label="📉 **Problemas de Valor Inicial**")
            st.caption("Valor Inicial, EDOS")
        with st.container(border=True):
            st.page_link("pages/[6] Integração Numérica.py", label="📈 **Integração Numérica**")
            st.caption("Integração através de métodos numéricos")
        with st.container(border=True):
            st.page_link("pages/[5] Mínimos Quadrados.py", label="📉 **Mínimos Quadrados**")
            st.caption("Cálculo de Mínimos Quadrados")

    st.markdown("---")

    st.header("Descrição do Projeto: ")

    st.write("""
    O projeto **Numérico Anônimos** foi desenvolvido com o objetivo de consolidar os conceitos adquiridos durante o semestre de Cálculo Numérico, ao mesmo tempo em que oferece uma ferramenta prática para a aplicação dos métodos matemáticos aprendidos. Este projeto visa ser uma contribuição útil e acessível para estudantes e profissionais que desejam entender e aplicar os métodos do Cálculo Numérico de forma simples e interativa.

    ### Objetivos

    - Criar uma ferramenta intuitiva e prática para resolver problemas de Cálculo Numérico.
    - Aplicar métodos fundamentais como **sistemas lineares**, **aproximação de raízes polinomiais**, **integração** e **derivação**.
    - Fornecer uma solução acessível, especialmente para quem está aprendendo Cálculo Numérico.

    ### Tecnologias Utilizadas

    - **Python**: Linguagem principal para o desenvolvimento da aplicação.
    - **Bibliotecas Python**: Uso de bibliotecas como `Numpy`, `Pandas` e `plotly.express` para implementação dos métodos matemáticos e visualização dos resultados.
    - **Streamlit**: Framework utilizado para criar o site interativo.

    ### Funcionalidades

    - **Métodos Matemáticos**: A aplicação oferece a implementação dos principais métodos de Cálculo Numérico, incluindo:
    
    - **Interface Interativa**: Utilização de uma interface simples e clara para facilitar a interação do usuário com a ferramenta.

    ### Contribuições e Impacto

    O projeto não apenas serviu para consolidar os conhecimentos adquiridos ao longo do semestre, mas também propõe uma ferramenta útil para qualquer pessoa que esteja interessada em aprender e aplicar os conceitos de Cálculo Numérico. A combinação de uma interface amigável com a funcionalidade do chatbot torna a experiência de aprendizado mais dinâmica e interativa.

    O **Numérico Anônimos** representa uma forma prática e acessível de aplicar métodos matemáticos fundamentais, promovendo o entendimento e o uso desses conceitos em problemas reais e desafiadores.
    """)

    st.markdown("---")
    st.title("Métodos Utilizados")

    st.markdown("<small>Métodos para aproximar raizes de polinômios:</small>", unsafe_allow_html=True)
    if st.button("**Método da Bisseção**", use_container_width=True):
        st.session_state["Bissecao"] = True  # Atualiza o estado para marcar a página como clicada
        st.rerun()  # Garante que a página seja recarregada com a nova página

    if st.button("**Método de Newton**", use_container_width=True):
        st.session_state["Newton"] = True  # Atualiza o estado para marcar a página como clicada
        st.rerun()  # Garante que a página seja recarregada com a nova página

    if st.button("**Método da Secante**", use_container_width=True):
        st.session_state["Secante"] = True  # Atualiza o estado para marcar a página como clicada
        st.rerun()  # Garante que a página seja recarregada com a nova página

    st.markdown("<small>Métodos para resolver Sistemas Lineares:</small>", unsafe_allow_html=True)
    if st.button("**Sistemas Lineares**", use_container_width=True):
        st.session_state["sistemas_lineares"] = True  # Atualiza o estado para marcar a página como clicada
        st.rerun()  # Garante que a página seja recarregada com a nova página

    st.markdown("<small>Métodos de Interpolação:</small>", unsafe_allow_html=True)
    if st.button("**Interpolação**", use_container_width=True):
        st.session_state["interpol"] = True  # Atualiza o estado para marcar a página como clicada
        st.rerun()  # Garante que a página seja recarregada com a nova página

    st.markdown("<small>Métodos dos Mínimos Quadrados:</small>", unsafe_allow_html=True)
    if st.button("**Mínimos Quadrados**", use_container_width=True):
        st.session_state["minimos"] = True  # Atualiza o estado para marcar a página como clicada
        st.rerun()  # Garante que a página seja recarregada com a nova página

    st.markdown("<small>Métodos de Integração:</small>", unsafe_allow_html=True)
    if st.button("**Integração**", use_container_width=True):
        st.session_state["integra"] = True  # Atualiza o estado para marcar a página como clicada
        st.rerun()  # Garante que a página seja recarregada com a nova página

    st.markdown("<small>Métodos para resolução de EDOs:</small>", unsafe_allow_html=True)
    if st.button("**Método de Euler**", use_container_width=True):
        st.session_state["edo1"] = True  # Atualiza o estado para marcar a página como clicada
        st.rerun()  # Garante que a página seja recarregada com a nova página

    if st.button("**Runge Kutta**", use_container_width=True):
        st.session_state["edo2"] = True  # Atualiza o estado para marcar a página como clicada
        st.rerun()  # Garante que a página seja recarregada com a nova página


