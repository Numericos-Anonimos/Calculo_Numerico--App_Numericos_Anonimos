import streamlit as st
from st_mathlive import mathfield
import sympy as sp
import re
from methods import bisseccao

# Função para extrair coeficientes do LaTeX
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
        st.error(f"Erro ao processar a equação: {e}")
        return [0, 0, 0]  # Retorna coeficientes nulos em caso de erro
    
    
# Função para limpar o LaTeX
def limpar_latex(latex):
    latex_limpo = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', latex)
    latex_limpo = latex_limpo.replace("^", "**")
    latex_limpo = latex_limpo.replace("x", "*x")
    latex_limpo = latex_limpo.replace(" ", "")
    return latex_limpo

# Função principal
def principal():
    if "calcular" in st.session_state and st.session_state["calcular"]:
        if "coeficientes" in st.session_state and st.session_state["coeficientes"]:
            st.header("Métodos de aproximação de raízes:")

            col1, col2, col3 = st.columns(3, gap='small')

            # Método da Bissecção
            if "metodo_bisseccao" in st.session_state and st.session_state["metodo_bisseccao"]:
                st.markdown("---")
                st.write("### Método escolhido: Bissecção")
                ini = st.number_input("Insira o limite inferior 'a' do intervalo", value=0)
                fim = st.number_input("Insira o limite superior 'b' do intervalo", value=0)
                tolerancia = st.number_input("Insira a tolerância", value=0.001)
                n = st.number_input("Insira o número 'n' máximo de interações", value=1)

                if st.button("Aplicar"):
                    coef = [float(c) for c in st.session_state["coeficientes"]] 
                    raiz, pontos = bisseccao(ini, fim, tolerancia, coef[2], coef[1], coef[0])
                    st.write(f"Raízes: {raiz} - Pontos: {pontos}")
                    
                    

            # Método de Newton
            elif "metodo_newton" in st.session_state and st.session_state["metodo_newton"]:
                st.markdown("---")
                st.write("### Método escolhido: Newton")
                x0 = st.number_input("Insira o valor inicial 'x0'", value=0.0)
                tolerancia = st.number_input("Insira a tolerância", value=0.001)
                n = st.number_input("Insira o número máximo de iterações 'n'", value=10)

                if st.button("Aplicar"):
                    pass  # Lógica do método de Newton

            # Método da Secante
            elif "metodo_secante" in st.session_state and st.session_state["metodo_secante"]:
                st.markdown("---")
                st.write("### Método escolhido: Secante")
                x0 = st.number_input("Insira o valor inicial 'x0'", value=0)
                x1 = st.number_input("Insira o valor inicial 'x1'", value=0)
                tolerancia = st.number_input("Insira a tolerância", value=0.001)
                n = st.number_input("Insira o número máximo de iterações 'n'", value=10)

                if st.button("Aplicar"):
                    pass  # Lógica do método da Secante

            # Coluna para os botões de escolha de métodos
            with col1:
                if st.button("**Método da Bissecção**"):
                    st.session_state["metodo_bisseccao"] = True
                    st.session_state["metodo_newton"] = False
                    st.session_state["metodo_secante"] = False
                    st.empty()
                    st.rerun()

            with col2:
                if st.button("**Método de Newton**"):
                    st.session_state["metodo_newton"] = True
                    st.session_state["metodo_bisseccao"] = False
                    st.session_state["metodo_secante"] = False
                    st.empty()
                    st.rerun()

            with col3:
                if st.button("**Método da Secante**"):
                    st.session_state["metodo_secante"] = True
                    st.session_state["metodo_bisseccao"] = False
                    st.session_state["metodo_newton"] = False
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
