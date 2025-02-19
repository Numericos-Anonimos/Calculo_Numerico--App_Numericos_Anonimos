import pandas as pd
from datetime import datetime
import streamlit as st
from PIL import Image
import plotly.express as px
import numpy as np

st.session_state['current_page'] = "Home"

st.logo(
    "src/img/unifesp_icon.ico",
    link="https://portal.unifesp.br/",
    icon_image="src/img/unifesp_icon.ico",
)

def bisseccao(f, ini, fim):
        if f(ini) * f(fim) >= 0:
            raise ValueError("Não há raiz no intervalo [a, b].")

        pontos = []
        while abs(fim - ini) > 1e-9:
            meio = (ini + fim) / 2.0
            pontos.append(meio)

            if f(ini) * f(meio) < 0:
                fim = meio
            else:
                ini = meio

        return meio, np.array(pontos)

    # Função para gerar o gráfico
def plotar_bisseccao(f, ini, fim, raiz, pontos, xmin, xmax, n):
    x = np.linspace(xmin, xmax, n)
    y = f(x)

    fig = px.line(x=x, y=y, title='',
                    labels={'x': 'x', 'y': 'f(x)'},
                    template='plotly_dark')

        # Adiciona os pontos intermediários do método
    fig.add_scatter(x=pontos, y=[f(p) for p in pontos],
                        mode='markers+lines', marker=dict(color='cyan', size=8),
                        line=dict(color='gray', dash='dot'), name='Passos da Bissecção')

        # Adiciona o ponto final da raiz
    fig.add_scatter(x=[raiz], y=[0], mode='markers',
                        marker=dict(color='red', size=12), name=f'Raiz Final: {raiz:.9f}')

        # Adiciona as linhas verticais delimitando o intervalo inicial
    fig.add_vline(x=ini, line_dash="dash", line_color="green", annotation_text="Início (a)")
    fig.add_vline(x=fim, line_dash="dash", line_color="green", annotation_text="Fim (b)")

    return fig







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

    if st.button("Voltar"):
        st.session_state["Bissecao"] = False
        st.empty()
        st.rerun()
elif "Secante" in  st.session_state and st.session_state["Secante"]:
    def read_file():
        with open('resumos/[ 1 ] Método das Secantes.md', 'r', encoding="utf-8") as file:
            data = file.read()
        st.write(data)

    read_file()

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

    if st.button("Voltar"):
        st.session_state["interpol"] = False
        st.empty()
        st.rerun()
elif "integra" in  st.session_state and st.session_state["integra"] :
    def read_file():
        with open('resumos/[ 7 ] Integração.md', 'r', encoding="utf-8") as file:
            data = file.read()
        st.write(data)

    read_file()

    if st.button("Voltar"):
        st.session_state["integra"] = False
        st.empty()
        st.rerun()
elif "edo1" in  st.session_state and st.session_state["edo1"] :
    def read_file():
        with open('resumos/[ 8 ] Método de Euler.md', 'r', encoding="utf-8") as file:
            data = file.read()
        st.write(data)

    read_file()

    if st.button("Voltar"):
        st.session_state["edo1"] = False
        st.empty()
        st.rerun()
elif "edo2" in  st.session_state and st.session_state["edo2"] :
    def read_file():
        with open('resumos/[ 8 ] Runge Kutta.md', 'r', encoding="utf-8") as file:
            data = file.read()
        st.write(data)

    read_file()

    if st.button("Voltar"):
        st.session_state["edo1"] = False
        st.empty()
        st.rerun()
else:
    st.title("Numéricos Anônimos")
    # Exibir os botões se a página "Bisseção" não foi escolhida
    col1, col2 = st.columns(2, gap='small')

    with col1:
        with st.container(border=True):
            st.button("🔢 **Equações de Uma Variável**", key="polinomio")
            st.caption("Aproximação de polinômios")
        with st.container(border=True):
            st.button("📐 **Sistemas Lineares**", key="lineares")
            st.caption("Resolução Sistemas Lineares com Matrizes")
        with st.container(border=True):
            st.button("📊 **Interpolação**", key="interpolacao")
            st.caption("Interpolação")

    with col2:
        with st.container(border=True):
            st.button("📉 **Problemas de Valor Inicial**", key="derivacao")
            st.caption("Valor Inicial, EDOS")
        with st.container(border=True):
            st.button("📈 **Integração**", key="integracao")
            st.caption("Integração Numérica")

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
    st.header("Métodos Utilizados")

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

    st.markdown("<small>Métodos para resolver sistemas lineares:</small>", unsafe_allow_html=True)
    if st.button("**Sistemas Lineares**", use_container_width=True):
        st.session_state["sistemas_lineares"] = True  # Atualiza o estado para marcar a página como clicada
        st.rerun()  # Garante que a página seja recarregada com a nova página

    st.markdown("<small>Métodos de interpolação:</small>", unsafe_allow_html=True)
    if st.button("**Interpolação**", use_container_width=True):
        st.session_state["interpol"] = True  # Atualiza o estado para marcar a página como clicada
        st.rerun()  # Garante que a página seja recarregada com a nova página

    if st.button("**Integração**", use_container_width=True):
        st.session_state["integra"] = True  # Atualiza o estado para marcar a página como clicada
        st.rerun()  # Garante que a página seja recarregada com a nova página

    if st.button("**Método de Euler**", use_container_width=True):
        st.session_state["edo1"] = True  # Atualiza o estado para marcar a página como clicada
        st.rerun()  # Garante que a página seja recarregada com a nova página

    if st.button("**Runge Kutta**", use_container_width=True):
        st.session_state["edo2"] = True  # Atualiza o estado para marcar a página como clicada
        st.rerun()  # Garante que a página seja recarregada com a nova página