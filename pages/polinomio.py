import streamlit as st
from st_mathlive import mathfield
import sympy as sp
import re
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import pandas as pd


# Função para limpar o LaTeX
def limpar_latex(latex):
    latex_limpo = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', latex)
    latex_limpo = latex_limpo.replace("^", "**")
    latex_limpo = latex_limpo.replace(" ", "")
    return latex_limpo

def extrair_coeficientes(latex):
    try:
        # Limpar o LaTeX
        latex_limpo = limpar_latex(latex)

        # Usar sympy para converter o LaTeX em uma expressão simbólica
        x = sp.symbols('x')
        expressao = sp.sympify(latex_limpo)

        # Extrair os coeficientes usando sympy
        grau_maximo = expressao.as_poly().degree()
        coeficientes = [expressao.coeff(x, i) for i in range(grau_maximo + 1)]

        # Garantir que sempre teremos 3 coeficientes
        while len(coeficientes) < 3:
            coeficientes.append(0)  # Preenche com 0 se algum coeficiente estiver faltando

        return coeficientes
    except Exception as e:
        print(f"Erro ao processar a equação: {e}")
        return [0, 0, 0]  # Retorna coeficientes nulos em caso de erro





def bisseccao(ini, fim, a, b, c, max_iter):
    # Função quadrática f(x) = ax^2 + bx + c
    def func(x):
        return a * (x ** 2) + b * x + c

    # Verificação se a função muda de sinal nos extremos
    if func(ini) * func(fim) >= 0:
        raise ValueError("Não há raiz no intervalo fornecido.")

    pontos = []
    iteracao = 0

    # Enquanto o intervalo for maior que a tolerância e o número máximo de iterações não for atingido
    while abs(fim - ini) > 1e-9 and iteracao < max_iter:
        meio = (ini + fim) / 2.0  # Ponto médio
        pontos.append(meio)

        # Verifica se a raiz está no intervalo [ini, meio]
        if func(ini) * func(meio) < 0:
            fim = meio
        else:
            ini = meio

        iteracao += 1

    return meio, np.array(pontos)



def plotar_bisseccao(ini, fim, raiz, pontos, xmin, xmax, n, a, b, c):
    def func(x):
        return a * (x ** 2) + b * x + c

    # Gera os pontos para a curva da função
    x_vals = np.linspace(xmin, xmax, n)
    y_vals = func(x_vals)

    # Cria o gráfico da função usando Plotly Express
    fig = px.line(
        x=x_vals,
        y=y_vals,
        title='Método da Bisseção',
        labels={'x': 'x', 'y': 'f(x)'},
        template='plotly_dark'
    )

    # Adiciona os pontos e as linhas dos passos da bissecção
    fig.add_scatter(
        x=pontos,
        y=[func(p) for p in pontos],
        mode='markers+lines',
        marker=dict(color='cyan', size=8),
        line=dict(color='gray', dash='dot'),
        name='Passos da Bissecção'
    )

    # Adiciona o ponto da raiz final (onde f(x) = 0)
    fig.add_scatter(
        x=[raiz],
        y=[0],
        mode='markers',
        marker=dict(color='red', size=12),
        name=f'Raiz Final: {raiz:.9f}'
    )

    # Adiciona linhas verticais para indicar os limites iniciais e finais do intervalo
    fig.add_vline(
        x=ini,
        line_dash="dash",
        line_color="green",
        annotation_text="Início (a)"
    )
    fig.add_vline(
        x=fim,
        line_dash="dash",
        line_color="green",
        annotation_text="Fim (b)"
    )

    return fig



def secante(x0, x1, tol, a, b, c, max_iter=100):
    pontos = []
    
    # Definindo a função quadrática
    def func(x):
        return a * (x ** 2) + b * x + c  # f(x) = ax^2 + bx + c
    
    # Verificação da diferença mínima entre x0 e x1
    if abs(x1 - x0) < 1e-6:
        raise ValueError("Os valores iniciais x0 e x1 são muito próximos. Ajuste-os para continuar.")
    
    for i in range(max_iter):
        f0 = func(x0)
        f1 = func(x1)
        
        # Evita a divisão por zero ou uma diferença muito pequena
        if abs(f1 - f0) < 1e-10:
            # Adiciona um pequeno valor para evitar divisão por zero
            x2 = x1 - f1 * (x1 - x0) / (f1 + 1e-10)
        else:
            # Fórmula da secante normal
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        
        pontos.append(x2)
        
        print(f"Iteração {i + 1}: x0 = {x0}, x1 = {x1}, x2 = {x2}, f(x0) = {f0}, f(x1) = {f1}")  # Depuração
        
        # Atualizando os valores para a próxima iteração
        x0, x1 = x1, x2
        
        # Critério de parada se a função for suficientemente pequena
        if abs(func(x1)) < tol:
            break
    
    return x1, pontos  # Retorna a raiz aproximada e os pontos intermediários


import numpy as np
import pandas as pd
import plotly.express as px

def plot_secantes(raiz, pontos, xmin, xmax, n, a, b, c):

    # Define a função quadrática
    def f(x):
        return a * (x ** 2) + b * x + c  # f(x) = ax² + bx + c

    # Gera os pontos para a curva da função
    x = np.linspace(xmin, xmax, n)
    y = f(x)
    df_func = pd.DataFrame({'x': x, 'f(x)': y})
    
    # Converte pontos para um array NumPy para operações vetorizadas
    pontos = np.array(pontos)
    df_pontos = pd.DataFrame({
        'x': pontos,
        'y': f(pontos),
        'Iterações': range(len(pontos))
    })

    # Cria o gráfico da função
    fig = px.line(df_func, x='x', y='f(x)', 
                  title='Método das Secantes', 
                  labels={'x': 'x', 'f(x)': 'f(x)'}, 
                  template='plotly_dark')

    # Adiciona os pontos das iterações conectados por linhas
    fig.add_scatter(x=df_pontos['x'], y=df_pontos['y'], mode='lines', 
                    marker=dict(color='cyan', size=8), name='Iterações')

    # Adiciona os "chutes" iniciais, verificando se há pontos suficientes
    if len(pontos) > 0:
        fig.add_scatter(x=[pontos[0]], y=[f(pontos[0])], mode='markers', 
                        marker=dict(color='orange', size=12, symbol='circle'),
                        name='Chute inicial (x0)')
    if len(pontos) > 1:
        fig.add_scatter(x=[pontos[1]], y=[f(pontos[1])], mode='markers', 
                        marker=dict(color='orange', size=12, symbol='diamond'),
                        name='Chute inicial (x1)')

    # Adiciona o ponto final (raiz aproximada)
    fig.add_scatter(x=[raiz], y=[f(raiz)], mode='markers',
                    marker=dict(color='yellow', size=12, symbol='star-diamond'),
                    name='Raiz Aproximada')

    # Se desejar traçar as secantes entre os pontos, descomente o trecho abaixo:
    """
    for i in range(len(pontos)-1):
        x_line = [pontos[i], pontos[i+1]]
        y_line = [f(pontos[i]), f(pontos[i+1])]
        fig.add_scatter(x=x_line, y=y_line, mode='lines', 
                        line=dict(color='red', dash='dash'), name='Secantes')
    """

    return fig




def derivada(a, b, x):
    return 2 * a * x + b  # Derivada da função quadrática f(x) = ax^2 + bx + c

def newton(x0, max_iter, a, b, c):
    pontos = []
    f = lambda x: a * (x ** 2) + b * x + c  # Função quadrática f(x) = ax^2 + bx + c

    for _ in range(max_iter):
        fx0 = f(x0)  # Avalia a função no ponto x0
        dfx0 = derivada(a, b, x0)  # Derivada da função quadrática no ponto x0

        if abs(fx0) < 1e-9:  # Critério de parada se a função estiver próxima de zero
            return x0, pontos

        if dfx0 == 0:  # Evita divisão por zero
            raise ValueError("Erro: Derivada igual a zero. Método falhou.")

        x1 = x0 - fx0 / dfx0  # Fórmula de Newton
        pontos.append(x1)

        x0 = x1  # Atualiza x0 para a próxima iteração

    return x0, pontos  # Retorna a raiz aproximada e os pontos intermediários


import numpy as np
import pandas as pd
import plotly.express as px

def plot_heron(raiz, pontos, xmin, xmax, n, a, b, c):
    # Define a função quadrática
    f = lambda x: a * (x ** 2) + b * x + c

    # Gera os valores de x para plotar a função e calcula os correspondentes y
    x_vals = np.linspace(xmin, xmax, n)
    y_vals = f(x_vals)
    
    # Cria DataFrames para a função f(x) e para a linha y = 0
    df_func = pd.DataFrame({'x': x_vals, 'y': y_vals, 'trace': 'f(x)'})
    df_zero = pd.DataFrame({'x': x_vals, 'y': np.zeros_like(x_vals), 'trace': 'y = 0'})
    
    # Converte os pontos de iteração para array, se necessário, e cria o DataFrame
    pontos = np.array(pontos)
    df_pontos = pd.DataFrame({'x': pontos, 'y': f(pontos), 'trace': 'Iterações'})
    
    # Concatena os DataFrames para gerar as linhas do gráfico
    df_lines = pd.concat([df_func, df_zero])
    
    # Cria o gráfico usando Plotly Express
    fig = px.line(
        df_lines,
        x='x',
        y='y',
        color='trace',
        title='Método de Heron (Newton)',
        labels={'x': 'x', 'y': 'f(x)'},
        template='plotly_dark'
    )
    
    # Adiciona os pontos das iterações como scatter
    fig.add_scatter(
        x=df_pontos['x'],
        y=df_pontos['y'],
        mode='markers',
        name='Iterações',
        marker=dict(color='cyan')
    )
    
    # Adiciona o ponto final (raiz) com destaque
    fig.add_scatter(
        x=[raiz],
        y=[f(raiz)],
        mode='markers',
        marker=dict(color='yellow', size=12, symbol='diamond'),
        name='Ponto Final'
    )
    
    return fig













# Função principal
def principal():
    graf = go.Figure()
    
    if "calcular" in st.session_state and st.session_state["calcular"]:
        if "coeficientes" in st.session_state and st.session_state["coeficientes"]:
            st.header("Métodos de aproximação de raízes:")

            col1, col2, col3 = st.columns(3, gap='small')

            # Método da Bissecção
            if "metodo_bisseccao" in st.session_state and st.session_state["metodo_bisseccao"]:
                st.markdown("---")
                st.write("### Método escolhido: Bissecção")

                ini = st.number_input("Insira o limite inferior do intervalo", value=0.0)
                fim = st.number_input("Insira o limite superior do intervalo", value=1.0)
                max_iter = st.number_input("Insira o número máximo de iterações", value=100, step=1)

                col_aplicar, col_voltar = st.columns([0.2, 0.2])

                with col_aplicar:
                    if st.button("Aplicar"):
                        coef = coef = [float(c) for c in st.session_state["coeficientes"]]
                        try:
                            raiz, pontos = bisseccao(ini, fim, coef[2], coef[1], coef[0], max_iter)
                            graf = plotar_bisseccao(ini, fim, raiz, pontos, ini-1, fim+1, max_iter, coef[2], coef[1], coef[0])
                            coef_2 = f"{coef[2]:.1f}"  # Formatação para 1 casa decimal
                            coef_1 = f"{coef[1]:+.1f}"  # Formatação para 1 casa decimal, incluindo o sinal
                            coef_0 = f"{coef[0]:+.1f}"  # Formatação para 1 casa decimal, incluindo o sinal

                            # Construção da equação com o formato correto
                            equacao = f"f(x) = {coef_2}x² {coef_1}x {coef_0}"
                            st.session_state["equacao"] = equacao
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
                        coef = [float(c) for c in st.session_state["coeficientes"]]
                        # Aqui vai a lógica do método de Newton, substitua o "pass" pelo cálculo do método
                        try:
                            raiz, pontos = newton(x0, n, coef[2], coef[1], coef[0])  # Supondo que 'metodo_newton' seja a função do método
                            graf = plot_heron(raiz,pontos, x0-2, x0+2, n, coef[2], coef[1], coef[0])
                            coef_2 = f"{coef[2]:.1f}"  # Formatação para 1 casa decimal
                            coef_1 = f"{coef[1]:+.1f}"  # Formatação para 1 casa decimal, incluindo o sinal
                            coef_0 = f"{coef[0]:+.1f}"  # Formatação para 1 casa decimal, incluindo o sinal

                            # Construção da equação com o formato correto
                            equacao = f"f(x) = {coef_2}x² {coef_1}x {coef_0}"
                            st.session_state["equacao"] = equacao
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
                coef = [float(c) for c in st.session_state["coeficientes"]]
                
                col4, col5 = st.columns([1, 1])  # Criando colunas para alinhar os botões

                with col4:
                    if st.button("Aplicar", use_container_width=True):
                        try:
                            # Garantindo que os coeficientes são passados corretamente
                            a, b, c = coef[2], coef[1], coef[0]  # Coeficientes da equação quadrática

                            # Aplicando o método da secante
                            raiz, pontos = secante(x0, x1, coef[2], a, b, c, n)
                            graf = plot_secantes(raiz, pontos, x0-1, x1+1, n, a, b, c)

                            # Formatando a equação para exibição
                            coef_2 = f"{coef[2]:.1f}"  # Formatação para 1 casa decimal
                            coef_1 = f"{coef[1]:+.1f}"  # Formatação para 1 casa decimal, incluindo o sinal
                            coef_0 = f"{coef[0]:+.1f}"  # Formatação para 1 casa decimal, incluindo o sinal

                            # Construção da equação com o formato correto
                            equacao = f"f(x) = {coef_2}x² {coef_1}x {coef_0}"
                            st.session_state["equacao"] = equacao
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
                    st.write(f"Equação dada: {st.session_state['equacao']}")
                    st.write(f"Raiz encontrada: {st.session_state['raiz']}")
                    st.write(f"Pontos intermediários: {st.session_state['pontos']}")
                    st.plotly_chart(st.session_state['grafico'])



            
            if "metodo_newton" in st.session_state and st.session_state["metodo_newton"] and "encontrou_resultado" in st.session_state and st.session_state["encontrou_resultado"]:
                st.markdown("---")
                st.write("### Resolvendo por: newton")
                if "raiz" in st.session_state:
                    st.write(f"Equação dada: {st.session_state['equacao']}")
                    st.write(f"Raiz encontrada: {st.session_state['raiz']}")
                    st.write(f"Pontos intermediários: {st.session_state['pontos']}")
                    st.plotly_chart(st.session_state['grafico'])




            if "metodo_secante" in st.session_state and st.session_state["metodo_secante"] and "encontrou_resultado" in st.session_state and st.session_state["encontrou_resultado"]:
                st.markdown("---")
                st.write("### Resolvendo por: secante")
                if "raiz" in st.session_state:
                    st.write(f"Equação dada: {st.session_state['equacao']}")
                    st.write(f"Raiz encontrada: {st.session_state['raiz']}")
                    st.write(f"Pontos intermediários: {st.session_state['pontos']}")
                    st.plotly_chart(st.session_state['grafico'])




            # Coluna para os botões de escolha de métodos
            with col1:
                if st.button("**Método da Bissecção**"):
                    st.session_state["metodo_bisseccao"] = True
                    st.session_state["metodo_newton"] = False
                    st.session_state["metodo_secante"] = False
                    st.session_state["encontrou_resultado"] = False
                    st.empty()
                    st.rerun()

            with col2:
                if st.button("**Método de Newton**"):
                    st.session_state["metodo_newton"] = True
                    st.session_state["metodo_bisseccao"] = False
                    st.session_state["metodo_secante"] = False
                    st.session_state["encontrou_resultado"] = False
                    st.empty()
                    st.rerun()

            with col3:
                if st.button("**Método da Secante**"):
                    st.session_state["metodo_secante"] = True
                    st.session_state["metodo_bisseccao"] = False
                    st.session_state["metodo_newton"] = False
                    st.session_state["encontrou_resultado"] = False
                    st.empty()
                    st.rerun()

            st.markdown("""
            <style>
            .stButton>button {
                width: 290px;
                height: 50px;
                font-size: 20px;
                display: block;
                margin-left: auto;
                margin-right: auto;
            }
            </style>
            """, unsafe_allow_html=True)

    else:
        st.title("Aproximação de Polinômios")
        st.write("Insira a equação no seguinte formato:")
        st.latex(r"ax^2 + bx + c")
        latex_input = mathfield(title="", value=r"", mathml_preview=True)

        if st.button("Calcular"):
            if latex_input:
                latex_input_str = latex_input[0]
                try:
                    coeficientes = extrair_coeficientes(latex_input_str)
                    st.session_state["coeficientes"] = coeficientes
                    st.session_state["calcular"] = True
                    st.rerun()
                except Exception as e:
                    st.write(f"Erro ao processar a equação: {e}")
            else:
                st.write("Por favor, insira um polinômio.")

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

principal()
