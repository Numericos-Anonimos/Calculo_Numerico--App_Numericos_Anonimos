import streamlit as st
from st_mathlive import mathfield
from PIL import Image
import re
import sympy as sp

st.set_page_config(layout="wide")
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
        return coeficientes
    except Exception as e:
        st.error(f"Erro ao processar a equação: {e}")
        return []

def limpar_latex(latex):
    # Remove formatação como \mathrm, \text, etc.
    latex_limpo = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', latex)  # Remove \mathrm{...}, \text{...}, etc.
    
    # Substituir "^" por "**" para compatibilidade com Python
    latex_limpo = latex_limpo.replace("^", "**")
    
    # Substituir "x" por "*x" para garantir multiplicação explícita
    latex_limpo = latex_limpo.replace("x", "*x")
    
    # Garantir que a expressão esteja em um formato compatível
    latex_limpo = latex_limpo.replace(" ", "")  # Remover espaços em branco extras
    
    return latex_limpo

im = Image.open("src/img/unifesp_icon.ico")

st.logo(
    im,
    link="https://portal.unifesp.br/",
    icon_image=im,
)


def principal():
    if "bisseccao" in st.session_state:
        st.session_state.clear()  # Limpa todos os estados armazenados no session_state
        st.experimental_rerun() 

        # Coletando os parâmetros (a, b, h) para o método da Bissecção
        st.markdown("### Insira os parâmetros:")
        st.session_state["a"] = st.number_input("Digite o valor de a:", value=0.0)
        st.session_state["b"] = st.number_input("Digite o valor de b:", value=0.0)
        st.session_state["h"] = st.number_input("Digite o valor de h:", value=0.0)

        if st.button("Submeter Parâmetros"):
            st.write(f"Parâmetros inseridos: a = {st.session_state['a']}, b = {st.session_state['b']}, h = {st.session_state['h']}")
            # Aqui você pode chamar a função para aplicar o método selecionado, por exemplo:
            # calcular_raiz(st.session_state["a"], st.session_state["b"], st.session_state["h"])
            st.session_state["metodo"] = None  # Limpa o método após submeter os parâmetros

    elif "newton" in st.session_state:
        pass  # Adicionar código para o método de Newton se necessário

    elif "secante" in st.session_state:
        pass  # Adicionar código para o método da Secante se necessário

    elif "calcular" in st.session_state and st.session_state["calcular"]:
        if "coeficientes" in st.session_state and st.session_state["coeficientes"]:
            st.header("Métodos de aproximação de raízes:")

            col1, col2, col3 = st.columns(3, gap='small')

            with col1:
                if st.button("**Método da Bissecção**\n\nO método da bissecção é um algoritmo numérico para encontrar raízes de funções contínuas. Baseia-se no Teorema do Valor Intermediário, que garante que, se uma função contínua $f(x)$ muda de sinal em um intervalo $[a, b]$, então existe pelo menos uma raiz nesse intervalo."):
                    st.session_state["metodo"] = "bisseccao"

            with col2:
                if st.button("# **Método de Newton**\n\nO método de Heron / Newton é um algoritmo iterativo utilizado para calcular a raiz quadrada de um número. Também conhecido como método da média aritmética ou método de aproximação sucessiva, ele se baseia na ideia de que uma aproximação para $\\sqrt{S}$ pode ser melhorada iterativamente, tomando a média entre um palpite e o quociente do número $S$ pelo palpite."):
                    st.session_state["metodo"] = "newton"

            with col3:
                if st.button("# **Método da Secante**\n\nO método das secantes é um algoritmo numérico iterativo para encontrar raízes de funções. Ele utiliza a aproximação da derivada por meio da inclinação da reta secante que passa por dois pontos da função, eliminando a necessidade de calcular a derivada explicitamente, como é feito no método de Newton-Raphson."):
                    st.session_state["metodo"] = "secante"

            if "metodo" in st.session_state:
                st.write(f"Método selecionado: {st.session_state['metodo']}")
                st.write("Coeficientes extraídos:", st.session_state["coeficientes"])
                st.rerun()
    
            st.markdown("""
            <style>
            .stButton>button {
                width: 290px;
                height: 350px;
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
