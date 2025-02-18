import streamlit as st
from st_mathlive import mathfield
from PIL import Image
import re
import sympy as sp

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

st.title("Aproximação de Polinômios")

st.write("Insira a equação no seguinte formato:")
st.latex(r"ax^2 + bx + c")

# Usando mathfield para entrada de LaTeX
latex_input = mathfield(title="Digite o polinômio:", value=r"", mathml_preview=True)

# Botão para calcular
if st.button("Calcular"):
    if latex_input:  # Verificando se há entrada
        latex_input_str = latex_input[0]  # Acessando a string do LaTeX
        
        #st.write(f"Equação inserida: {latex_input_str}")  # Exibir a entrada para depuração
        
        try:
            # Chamar a função para extrair coeficientes
            coeficientes = extrair_coeficientes(latex_input_str)
            
            if coeficientes:
                # Exibir os coeficientes
                st.write("Coeficientes extraídos:", coeficientes)
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
