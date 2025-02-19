import pandas as pd
from datetime import datetime
import streamlit as st
from PIL import Image
import plotly.express as px
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
                  title='',
                  labels={'x': 'x', 'y': 'f(x)'},
                  template='plotly_dark')
    
    fig.add_scatter(x=df_pontos['x'], y=df_pontos['y'],
                    mode='markers', name='Iterações',
                    marker=dict(color='cyan'))
    
    fig.add_scatter(x=[raiz], y=[f(raiz)], mode='markers', 
                    marker=dict(color='yellow', size=12, symbol='diamond'),
                    name='Ponto Final')
    
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
    
    df_func = pd.DataFrame({'x': x, 'f(x)': y})
    df_pontos = pd.DataFrame({'x': pontos, 'y': f(pontos), 'Iterações': range(len(pontos))})

    fig = px.line(df_func, x='x', y='f(x)', title='', 
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


def jacobi_solver(A, b, x0, tol=1e-6, max_iter=100):
    n = len(b)
    x = x0.copy()
    iterations = [x.copy()]
    for k in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
        iterations.append(x_new.copy())
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
    return x, iterations

# Função de plotagem
def plot_jacobi(iter_jacobi):
    df = pd.DataFrame(iter_jacobi, columns=['x1', 'x2'])
    df['Iteração'] = np.arange(len(df))
    df_melted = df.melt(id_vars='Iteração', value_vars=['x1', 'x2'], var_name='Variável', value_name='Valor')
    
    fig = px.line(df_melted, x='Iteração', y='Valor', color='Variável', markers=True,
                  title="", template="plotly_dark")
    return fig

def gauss_seidel_solver(A, b, x0, tol=1e-6, max_iter=100):
    n = len(b)
    x = x0.copy()
    iterations = [x.copy()]
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i))
            s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i, i]
        iterations.append(x.copy())
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            break
    return x, iterations

def criar_dataframe(iterations, metodo):
    data = []
    for k, x in enumerate(iterations):
        for i, xi in enumerate(x):
            data.append({
                "Iteração": k, "Variável": f"$x_{{{i+1}}}$",
                "Valor": xi, "Método": metodo
            })
    return pd.DataFrame(data)

def polynomial(x, coef):
    result = 0
    degree = len(coef) - 1
    for i in range(len(coef)):
        result += coef[i] * x ** (degree - i)
    return result

def lagrange(x_pontos, y_pontos, epsilon=1e-10):
    n = len(x_pontos)
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            A[i, j] = x_pontos[i] ** (n - j - 1)

    coef = np.linalg.solve(A, y_pontos)
    coef = np.array([0 if abs(c) < epsilon else c for c in coef])

    coef_arredondados = [round(c, 4) for c in coef]

    return coef

def plotar_polinomio(x_pontos, y_pontos, coef, xmin, xmax, n):
    x = np.linspace(xmin, xmax, n)
    y = polynomial(x, coef)

    df_pontos = pd.DataFrame({'x': x_pontos, 'y': y_pontos})
    df_polynomial = pd.DataFrame({'x': x, 'y': y})

    fig = px.scatter(df_pontos, x="x", y="y",
                    title="",
                    labels={"x": "x", "y": "y"},
                    template="plotly_dark",
                    color_discrete_sequence=["cyan"])

    fig.add_traces(px.line(df_polynomial, x="x", y="y",
                            template="plotly_dark",
                            color_discrete_sequence=["yellow"]).data)

    return fig




def a(x, y):
    n, s1, s2, s3, s4 = len(x), sum(x*y), sum(x), sum(y), sum(x**2)
    return (n * s1 - s2 * s3) / (n * s4 - s2**2)

def b(x, y, a):
    return np.mean(y) - a * np.mean(x)

def calcular_regressao_linear(x, a, b):
    return a * x + b

def plotar_regressao_linear(x, y, a, b, xmin, xmax, n):
    x_line = np.linspace(xmin, xmax, n)
    y_line = calcular_regressao_linear(x_line, a, b)

    df_points = pd.DataFrame({'x': x, 'y': y})
    df_line = pd.DataFrame({'x': x_line, 'y': y_line})

    fig = px.scatter(df_points, x='x', y='y',
                     title='Regressão Linear',
                     labels={'x': 'Tempo (minutos)', 'y': 'Temperatura'},
                     template='plotly_dark',
                     color_discrete_sequence=['cyan'])

    fig.add_traces(px.line(df_line, x='x', y='y', color_discrete_sequence=['yellow']).data)
    
    return fig 

def calcular_coeficientes(x, y):
    """Calcula os coeficientes da regressão linear (a e b)."""
    n, s1, s2, s3, s4 = len(x), sum(x*y), sum(x), sum(y), sum(x**2)
    a = (n * s1 - s2 * s3) / (n * s4 - s2**2)
    b = np.mean(y) - a * np.mean(x)
    return a, b

def calcular_regressao_linear(x, a, b):
    """Calcula os valores previstos pela regressão linear."""
    return a * x + b

def plotar_regressao_linear(x, y, a, b, xmin, xmax, n):
    """Cria um gráfico interativo da regressão linear."""
    x_line = np.linspace(xmin, xmax, n)
    y_line = calcular_regressao_linear(x_line, a, b)

    df_points = pd.DataFrame({'x': x, 'y': y})
    df_line = pd.DataFrame({'x': x_line, 'y': y_line})

    fig = px.scatter(df_points, x='x', y='y',
                     title='',
                     labels={'x': 'Tempo (minutos)', 'y': 'Temperatura'},
                     template='plotly_dark',
                     color_discrete_sequence=['cyan'])

    fig.add_traces(px.line(df_line, x='x', y='y', color_discrete_sequence=['yellow']).data)
    
    return fig 


def regressao_linear(x_vals, y_vals):
    """Calcula os coeficientes da regressão linear aplicada ao modelo exponencial."""
    n = len(x_vals)
    Sx = np.sum(x_vals)
    Sy = np.sum(y_vals)
    Sxx = np.sum(x_vals**2)
    Sxy = np.sum(x_vals * y_vals)
    b = (n * Sxy - Sx * Sy) / (n * Sxx - Sx**2)
    alpha = np.mean(y_vals) - b * np.mean(x_vals)

    return alpha, b

def modelo_exponencial(x_val, a, b):
    """Calcula os valores previstos pelo modelo exponencial."""
    return a * np.exp(b * x_val)

def plotar_modelo_exponencial(x, y, a, b, xmin, xmax, n):
    """Cria um gráfico interativo do modelo exponencial."""
    x_line = np.linspace(xmin, xmax, n)
    y_line = modelo_exponencial(x_line, a, b)

    df_points = pd.DataFrame({'x': x, 'y': y})
    df_line = pd.DataFrame({'x': x_line, 'y': y_line})

    fig = px.scatter(df_points, x='x', y='y',
                     title='',
                     labels={'x': 'x', 'y': 'y'},
                     template='plotly_dark',
                     color_discrete_sequence=['cyan'])

    fig.add_traces(px.line(df_line, x='x', y='y', color_discrete_sequence=['yellow']).data)
    
    return fig

def calcular_coeficientes2(x, y, grau):
    """Calcula os coeficientes do polinômio de grau especificado via mínimos quadrados."""
    n = len(x)
    A = np.zeros((n, grau + 1))
    for i in range(grau + 1):
        A[:, i] = x ** i

    AT_A = np.dot(A.T, A)
    AT_y = np.dot(A.T, y)
    coef = np.linalg.solve(AT_A, AT_y)
    return coef

def calcular_polinomio(x, coef):
    """Calcula os valores preditos pelo polinômio ajustado."""
    y_pred = np.zeros_like(x)
    for i, c in enumerate(coef):
        y_pred += c * (x ** i)
    return y_pred

def plotar_multiplos_polinomios_interativo(x, y, coefs_labels, xmin, xmax, n):
    """Gera um gráfico interativo comparando diferentes ajustes polinomiais."""
    df_points = pd.DataFrame({'x': x, 'y': y})
    fig = px.scatter(df_points, x='x', y='y',
                     title='',
                     labels={'x': 'x', 'y': 'y'},
                     template='plotly_dark',
                     color_discrete_sequence=['cyan'])
    
    x_line = np.linspace(xmin, xmax, n)
    for coef, label in coefs_labels:
        y_line = calcular_polinomio(x_line, coef)
        fig.add_scatter(x=x_line, y=y_line, 
                        mode='lines', 
                        name=label,
                        line=dict(width=2))
    
    fig.update_layout(legend_title_text='Polinômios',
                      hovermode='x unified')
    
    return fig


# Método de Euler
def euler(func, x0, y0, xf, h):
    n = int((xf - x0) / h)
    xs = np.linspace(x0, xf, n+1)
    ys = np.zeros(n+1)
    ys[0] = y0

    for i in range(n):
        ys[i+1] = ys[i] + h * func(xs[i], ys[i])

    return xs, ys

def plot_euler(func, x0, y0, xf, h):
    xs, ys = euler(func, x0, y0, xf, h)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode='lines+markers',
        name='Método de Euler',
        line=dict(color='lime', width=3),
        marker=dict(color='cyan', size=8)
    ))

    return fig

def f(x, y):
    return -2 * y

def runge_kutta_2(func, y0, t0, t_end, dt):
    t_values = np.arange(t0, t_end, dt)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        t = t_values[i-1]
        y = y_values[i-1]

        k1 = dt * func(t, y)
        k2 = dt * func(t + dt, y + k1)

        y_values[i] = y + (k1 + k2) / 2

    return t_values, y_values

# Método de Runge-Kutta de 4ª ordem (RK4)
def runge_kutta_4(func, y0, t0, t_end, dt):
    t_values = np.arange(t0, t_end, dt)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        t = t_values[i-1]
        y = y_values[i-1]

        k1 = dt * func(t, y)
        k2 = dt * func(t + dt/2, y + k1/2)
        k3 = dt * func(t + dt/2, y + k2/2)
        k4 = dt * func(t + dt, y + k3)

        y_values[i] = y + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t_values, y_values

def plot_runge_kutta(func, y0, t0, t_end, dt):
    # Obter as soluções numéricas
    t_rk2, y_rk2 = runge_kutta_2(func, y0, t0, t_end, dt)
    t_rk4, y_rk4 = runge_kutta_4(func, y0, t0, t_end, dt)
    
    # Solução exata (para f(t,y) = -2*y, temos y = y0 * exp(-2*t))
    t_exact = np.linspace(t0, t_end, 200)
    y_exact = y0 * np.exp(-2 * t_exact)
    
    # Criando o gráfico com subplots para comparar RK2 e RK4
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        subplot_titles=("Runge-Kutta de 2ª Ordem (RK2)", "Runge-Kutta de 4ª Ordem (RK4)"))
    
    # Adicionando o gráfico para RK2 na primeira linha
    fig.add_trace(go.Scatter(x=t_rk2, y=y_rk2, mode='lines+markers',
                             name='RK2',
                             line=dict(color='cyan', width=3),
                             marker=dict(size=8)),
                  row=1, col=1)
    # Adicionando a solução exata para RK2
    fig.add_trace(go.Scatter(x=t_exact, y=y_exact, mode='lines',
                             name='Solução Exata',
                             line=dict(color='yellow', dash='dash', width=2)),
                  row=1, col=1)
    
    # Adicionando o gráfico para RK4 na segunda linha
    fig.add_trace(go.Scatter(x=t_rk4, y=y_rk4, mode='lines+markers',
                             name='RK4',
                             line=dict(color='magenta', width=3),
                             marker=dict(size=8)),
                  row=2, col=1)
    # Adicionando a solução exata para RK4 (evitando legenda duplicada)
    fig.add_trace(go.Scatter(x=t_exact, y=y_exact, mode='lines',
                             name='Solução Exata',
                             line=dict(color='yellow', dash='dash', width=2),
                             showlegend=False),
                  row=2, col=1)
    
    
    # Atualizando os títulos dos subplots
    fig.update_xaxes(title_text="t (tempo)", row=2, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_yaxes(title_text="y", row=2, col=1)
    
    return fig

def trapezoidal_composta(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h * (y[0] + 2*np.sum(y[1:n]) + y[n]) / 2

def plot_trapezoidal_composta_interactive(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)

    xx = np.linspace(a, b, 1000 + n)
    yy = f(xx)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xx, y=yy, mode='lines', line=dict(color='blue', width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', marker=dict(color='red', size=8), line=dict(color='red', width=1), showlegend=False))

    for i in range(n):
        polygon_x = [x[i], x[i+1], x[i+1], x[i]]
        polygon_y = [y[i], y[i+1], 0, 0]
        fig.add_trace(go.Scatter(x=polygon_x, y=polygon_y, fill='toself', mode='none', fillcolor='rgba(255,165,0,0.4)', line=dict(color='black'), showlegend=False))

    fig.update_layout(template='plotly_dark', title_text='', xaxis_title='x', yaxis_title='f(x)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def simpson_composta(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("O número de subintervalos (n) deve ser par para a regra de Simpson composta.")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    soma_impar = np.sum(y[1:n:2])   # índices 1, 3, 5, ..., n-1
    soma_par   = np.sum(y[2:n-1:2]) # índices 2, 4, 6, ..., n-2

    return h / 3 * (y[0] + y[n] + 4 * soma_impar + 2 * soma_par)

def plot_simpson_composta_interactive(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("O número de subintervalos (n) deve ser par para a regra de Simpson composta.")

    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)

    xx = np.linspace(a, b, 1000)
    yy = f(xx)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=xx, y=yy,
                             mode='lines',
                             line=dict(color='blue', width=2),
                             showlegend=False))

    fig.add_trace(go.Scatter(x=x, y=y,
                             mode='lines+markers',
                             marker=dict(color='red', size=8),
                             line=dict(color='red', width=1),
                             showlegend=False))

    for i in range(0, n, 2):
        x_parabola = np.linspace(x[i], x[i+2], 100)
        coef = np.polyfit([x[i], x[i+1], x[i+2]], [y[i], y[i+1], y[i+2]], 2)
        y_parabola = np.polyval(coef, x_parabola)

        polygon_x = np.concatenate([x_parabola, x_parabola[::-1]]).tolist()
        polygon_y = np.concatenate([y_parabola, np.zeros_like(y_parabola)]).tolist()

        fig.add_trace(go.Scatter(x=polygon_x, y=polygon_y,
                                 fill='toself',
                                 mode='none',
                                 fillcolor='rgba(255,165,0,0.4)',
                                 line=dict(color='black'),
                                 showlegend=False))

    fig.update_layout(template='plotly_dark',
                      title_text='',
                      xaxis_title='x',
                      yaxis_title='f(x)',
                      plot_bgcolor='rgba(0,0,0,0)')
    return fig


def format_function_markdown(f, var_name='x'):
    x = sp.Symbol(var_name)
    expr = f(x)
    return f"$$ {sp.latex(expr)} $$"








