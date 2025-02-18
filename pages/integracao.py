import streamlit as st
from st_mathlive import mathfield
import sympy as sp
import re

def corrigir_sintaxe(latex):
    """Corrige a sintaxe do LaTeX para ser compatível com SymPy."""
    latex = latex.replace("\\,", "").replace(" ", "")  # Remover espaços e "\,"
    latex = latex.replace("^", "**")  # Substituir "^" por "**"

    # Remover comandos como \mathrm{...}
    latex = re.sub(r'\\mathrm{([^}]*)}', r'\1', latex)

    # Substituir `_a^b` por `{a}^{b}`
    latex = re.sub(r'\\int_([^\^]+)\^([^\s]+)', r'\\int_{\1}^{\2}', latex)

    # Remover qualquer `*` extra antes do limite superior
    latex = re.sub(r'(\d)\*\*', r'\1**', latex)  # Evita `0**1` no output

    # Adicionar "*" entre número e variável (ex.: "2x" → "2*x")
    latex = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', latex)

    return latex

def extrair_integral(latex):
    try:
        # **Corrigir a sintaxe**
        latex = corrigir_sintaxe(latex)

        # **Regex para capturar integrais no formato correto**
        padrao_integral = r'\\int_{([^}]+)}\^{([^}]+)}\s*([\w\+\-\*/\^\(\)]+)\s*dx'
        match = re.search(padrao_integral, latex)

        if not match:
            st.error("Formato incorreto da integral. Use \\int_{a}^{b} expressão dx")
            return None
        
        # **Extrair os limites inferior, superior e a função**
        lim_inferior, lim_superior, expressao = match.groups()

        # **Converter limites para números ou símbolos**
        lim_inferior = sp.sympify(lim_inferior) if lim_inferior else None
        lim_superior = sp.sympify(lim_superior) if lim_superior else None

        # **Definir variável simbólica**
        x = sp.symbols('x')

        # **Converter a expressão para SymPy**
        expressao_simbolica = sp.sympify(expressao, locals={'x': x})

        return lim_inferior, lim_superior, expressao_simbolica

    except Exception as e:
        st.error(f"Erro ao processar a equação: {e}")
        return None

# **Interface Streamlit**
st.title("Calculadora de Integrais")

st.write("Insira a integral no formato:")
st.latex(r"\int_{a}^{b} f(x) dx")

# **Entrada MathLive**
latex_input = mathfield(title="Digite a integral:", value=r"", mathml_preview=True)

# **Botão de calcular**
if st.button("Calcular"):
    if latex_input:
        latex_str = latex_input[0]  # Evitar erro de lista
        resultado = extrair_integral(latex_str)

        if resultado:
            lim_inf, lim_sup, func = resultado
            st.write(f"Limites: {lim_inf} a {lim_sup}")
            st.write(f"Função extraída: {func}")
    else:
        st.write("Por favor, insira uma integral válida.")