# Mínimos Quadrados
## Fundamentos

O método dos mínimos quadrados é uma técnica estatística utilizada  para ajustar um modelo aos dados, minimizando a soma dos quadrados dos resíduos (as diferenças entre os valores observados e os valores previstos pelo modelo). Ou seja, o objetivo é encontrar os parâmetros do modelo que minimizam o erro total de previsão.

Para um conjunto de $n$ observações $(x_i, y_i)$, a soma dos erros quadráticos é dada por:
$$S(\beta) = \sum_{i=1}^{n} \left( y_i - f(x_i, \beta) \right)^2,$$

onde $f(x_i, \beta)$ é o valor predito pelo modelo e $\beta$ representa os parâmetros a serem estimados.

---

## Estimativa dos Parâmetros
Em notação matricial, a solução analítica para os coeficientes do modelo, no contexto de regressão linear múltipla, é dada por:
$$\boldsymbol{\hat{\beta}} = (X^TX)^{-1}X^Ty,$$

onde:
- $x$ é a matriz de dados que inclui uma coluna de 1's (para o intercepto),
- $y$ é o vetor das observações,
- $\boldsymbol{\hat{\beta}}$ é o vetor de coeficientes estimados.

## Regressão Linear

Quando aplicado à regressão linear simples, o método dos mínimos quadrados encontra os coeficientes $\beta_0$ e $\beta_1$ que minimizam a soma dos quadrados dos resíduos:

$$y = \beta_0 + \beta_1 x + \varepsilon,$$

e os parâmetros $\beta_0$ e $\beta_1$ podem ser estimados através das seguintes fórmulas, que minimizam a soma dos quadrados dos resíduos:

$$\beta_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}, \quad \beta_0 = \bar{y} - \beta_1 \bar{x}.$$

### Código em Python
Suponha que temos um conjunto de pontos abaixo e desejamos ajustar um modelo linear:

| Tempo | 1:00 | 2:00 | 3:00 | 4:00 | 5:00 | 6:00| 7:00 |
|--------|------|------|------|------|------|------|------|
| Temperatura | 13 | 15 | 20 | 14 | 15 | 13 | 10 |


~~~python
import numpy as np
import pandas as pd
import plotly.express as px

def a (x, y):
    n, s1, s2, s3, s4 = len(x), sum(x*y), sum(x), sum(y), sum(x**2)
    return (n*s1 - s2*s3) / (n*s4 - s2**2)

def b (x, y, a):
    return np.mean(y) - a*np.mean(x)

def calcular_regressao_linear(x, a, b):
    return a * x + b

def plotar_regressao_linear(x, y, a, b, xmin, xmax, n):
    x_line = np.linspace(xmin, xmax, n)
    y_line = calcular_regressao_linear(x_line, a, b)

    df_points = pd.DataFrame({'x': x, 'y': y})
    df_line = pd.DataFrame({'x': x_line, 'y': y_line})

    fig = px.scatter(df_points, x='x', y='y',
                    title='Regressão Linear',
                    labels={'x': 'Tempo(minutos)', 'y': 'Temperatura'},
                    template='plotly_dark',
                    color_discrete_sequence=['cyan'])

    fig.add_traces(px.line(df_line, x='x', y='y', color_discrete_sequence=['yellow']).data)
    
    return fig 

x = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([13, 15, 20, 14, 15, 13, 10])
a_ = a(x, y)
b_ = b(x, y, a_)
fig = plotar_regressao_linear(x, y, a_, b_, 0, 8, 100)
fig.show()
~~~

<grafico>

## Modelos Exponenciais

O método dos mínimos quadrados também pode ser aplicado a modelos não lineares. Um exemplo bastante comum é a regressão exponencial, na qual o modelo é:
$$y = \alpha e^{\beta x}.$$

Para facilitar a estimação dos parâmetros usando mínimos quadrados, esse modelo pode ser linearizado tomando o logaritmo dos dois lados:

$$\ln(y) = \ln(\alpha) + \beta x.$$

Após essa transformação, o problema se reduz a ajustar uma regressão linear entre $\ln(y)$ e $x$, estimando $\ln(\alpha)$ e $\beta$. O parâmetro $\alpha$ pode ser recuperado pela exponenciação:

$$\alpha = e^{\ln(\alpha)}.$$

### Código em Python
Suponha que temos um conjunto de pontos abaixo e desejamos ajustar um modelo exponencial:

| **x** | 0.0  | 1.5  | 2.5  | 3.5  | 4.5  |
|-------|------|------|------|------|------|
| **y** | 2.0  | 3.6  | 5.4  | 8.1  | 12.0 |

~~~python
import numpy as np
import pandas as pd
import plotly.express as px

def regressao_linear(x_vals, y_vals):
    n = len(x_vals)
    Sx = np.sum(x_vals)
    Sy = np.sum(y_vals)
    Sxx = np.sum(x_vals**2)
    Sxy = np.sum(x_vals * y_vals)
    b = (n * Sxy - Sx * Sy) / (n * Sxx - Sx**2)
    alpha = np.mean(y_vals) - b * np.mean(x_vals)

    return alpha, b

def modelo_exponencial(x_val, a, b):
    return a * np.exp(b * x_val)

def plotar_modelo_exponencial(x, y, a, b, xmin, xmax, n):
    x_line = np.linspace(xmin, xmax, n)
    y_line = modelo_exponencial(x_line, a, b)

    df_points = pd.DataFrame({'x': x, 'y': y})
    df_line = pd.DataFrame({'x': x_line, 'y': y_line})

    fig = px.scatter(df_points, x='x', y='y',
                    title='Modelo Exponencial',
                    labels={'x': 'x', 'y': 'y'},
                    template='plotly_dark',
                    color_discrete_sequence=['cyan'])

    fig.add_traces(px.line(df_line, x='x', y='y', color_discrete_sequence=['yellow']).data)
    
    return fig

x = np.array([0.0, 1.5, 2.5, 3.5, 4.5])
y = np.array([2.0, 3.6, 5.4, 8.1, 12.0])
ln_y = np.log(y)

alpha, b = regressao_linear(x, ln_y)
a = np.exp(alpha)
fig = plotar_modelo_exponencial(x, y, a, b, -1, 6, 100)
fig.show()
~~~

<grafico>

## Modelo Polinomial

Podemos ajustar um modelo polinomial de grau $k$ aos dados, minimizando a soma dos quadrados dos resíduos utilizando a notação matricial:

$$\boldsymbol{\hat{\beta}} = (X^TX)^{-1}X^Ty,$$

### Código em Python
Suponha que temos um conjunto de pontos abaixo e desejamos ajustar modelos de grau 1 e 2:

| **x** | 0 | 1 | 2 | 3 | 4 | 5 | 
|-------|---|---|---|---|---|---|
| **y** | 3 | 15 | 47 | 99 | 171 | 263 |

~~~python
def calcular_coeficientes(x, y, grau):
    n = len(x)
    A = np.zeros((n, grau + 1))
    for i in range(grau + 1):
        A[:, i] = x ** i

    AT_A = np.dot(A.T, A)
    AT_y = np.dot(A.T, y)
    coef = np.linalg.solve(AT_A, AT_y)
    return coef

def calcular_polinomio(x, coef):
    y_pred = np.zeros_like(x)
    for i, c in enumerate(coef):
        y_pred += c * (x ** i)
    return y_pred

def calcular_erro(x, y, coef):
    y_pred = calcular_polinomio(x, coef)
    erro = np.sum((y - y_pred) ** 2)
    return erro

def plotar_multiplos_polinomios_interativo(x, y, coefs_labels, xmin, xmax, n):
    df_points = pd.DataFrame({'x': x, 'y': y})
    fig = px.scatter(df_points, x='x', y='y',
                     title='Comparação de Polinômios',
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


x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([3.0, 15.0, 47.0, 99.0, 171.0, 263.0])
coef_grau1 = calcular_coeficientes(x, y, 1)
coef_grau2 = calcular_coeficientes(x, y, 2)
fig = plotar_multiplos_polinomios_interativo(x, y, [(coef_grau1, 'Grau 1'), (coef_grau2, 'Grau 2')], 0, 5, 100)
fig.show()
~~~

