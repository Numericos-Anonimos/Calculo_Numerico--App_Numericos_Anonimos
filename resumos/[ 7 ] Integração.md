# Integração Numérica: Regras Trapezoidal e de Simpson

A integração numérica é uma ferramenta essencial quando não conseguimos (ou não queremos) encontrar a antiderivada exata de uma função ou quando os dados disponíveis vêm de medições experimentais. Ela consiste em aproximar a área sob a curva $f(x)$ em um intervalo $[a,b]$ por métodos que usam somas finitas de áreas de figuras geométricas.

A seguir, apresentamos uma aula completa sobre os métodos da regra trapezoidal, regra de Simpson e suas versões compostas, incluindo teoria, intuição e exemplos.

---

## 1. Introdução e Motivação

Quando temos uma integral definida:
$$
I = \int_{a}^{b} f(x)\,dx
$$
muitas vezes não é possível (ou é difícil) encontrar uma antiderivada analítica. Métodos numéricos aproximam essa área dividindo o intervalo $[a,b]$ em subintervalos e aproximando a função $f(x)$ por funções simples (retas ou polinômios de baixo grau) cuja integral é conhecida. Dessa forma, podemos obter uma boa aproximação para $I$.

---

## 2. Regras Compostas

Quando o intervalo $[a,b]$ é grande ou a função apresenta variações mais acentuadas, utilizar apenas um único trapézio ou parábola pode não ser suficiente para obter uma boa aproximação. Assim, subdividimos o intervalo em vários subintervalos e aplicamos o método em cada um, somando os resultados. Essas são as **regras compostas**.

### 2.1 Regra Trapezoidal Composta

Dividindo o intervalo $[a,b]$ em $n$ subintervalos iguais, com largura
$
h = \frac{b-a}{n},
$
os pontos de divisão são:
$$
x_0 = a,\quad x_1 = a+h,\quad \dots,\quad x_n = b.
$$
A fórmula composta é:
$$
I \approx \frac{h}{2} \left[f(x_0) + 2 \sum_{i=1}^{n-1} f(x_i) + f(x_n)\right]
$$
Essa abordagem melhora a precisão, pois a função é aproximada por segmentos lineares em pequenos intervalos.

---

### 2.2 Exemplo

$$
\frac{1}{4 + \sin(20x)}
$$

# Código em Python
~~~python

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def trapezoidal_composta(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h * (y[0] + 2*np.sum(y[1:n]) + y[n]) / 2

def plot_trapezoidal_composta_interactive(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)

    # Pontos densos para a curva real da função
    xx = np.linspace(a, b, 1000 + n)
    yy = f(xx)

    # Criação da figura com template dark
    fig = go.Figure()

    # Traço da função real (curva contínua)
    fig.add_trace(go.Scatter(x=xx, y=yy,
                             mode='lines',
                             line=dict(color='blue', width=2),
                             showlegend=False))

    # Traço dos pontos de aproximação e conexão entre eles
    fig.add_trace(go.Scatter(x=x, y=y,
                             mode='lines+markers',
                             marker=dict(color='red', size=8),
                             line=dict(color='red', width=1),
                             showlegend=False))

    # Desenhando os trapézios preenchidos
    for i in range(n):
        # Polígono do trapézio: vai do ponto (x[i], y[i]) a (x[i+1], y[i+1]) e volta para o eixo x (y=0)
        polygon_x = [x[i], x[i+1], x[i+1], x[i]]
        polygon_y = [y[i], y[i+1], 0, 0]
        fig.add_trace(go.Scatter(x=polygon_x, y=polygon_y,
                                 fill='toself',
                                 mode='none',
                                 fillcolor='rgba(255,165,0,0.4)',  # laranja com transparência
                                 line=dict(color='black'),
                                 showlegend=False))

    # Configuração final do layout
    fig.update_layout(template='plotly_dark',
                      title_text='Aproximação da Integral pela Regra dos Trapézios Composta',
                      xaxis_title='x',
                      yaxis_title='f(x)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show()

# Definição da função a integrar
f = lambda x:  1/(4 + np.sin(20*x))

r_trap = trapezoidal_composta(f, 0, 3, 30)
plot_trapezoidal_composta_interactive(f, 0, 3, 30)

~~~

<grafico>

---
### 2.3 Regra de Simpson Composta

Para a **regra de Simpson composta** é necessário que o número de subintervalos $n$ seja par (pois se utiliza uma parábola para cada dois subintervalos). Com $h = \frac{b-a}{n}$ e pontos $x_0, x_1, \dots, x_n$, a fórmula é:
$$
I \approx \frac{h}{3} \left[ f(x_0) + f(x_n) + 4 \sum_{i \, \text{ímpar}} f(x_i) + 2 \sum_{i \, \text{par, } i \neq 0,n} f(x_i) \right]
$$
Essa regra utiliza coeficientes diferentes para os pontos de índices ímpares e pares (exceto os extremos) e, em geral, fornece uma aproximação de alta precisão para funções suaves.

A regra de Simpson composta pode ser aplicada da mesma forma que a regra trapezoidal composta, mas tende a oferecer uma precisão superior ao integrar funções suaves, desde que o número de subintervalos seja escolhido de forma adequada (por exemplo, 10 subintervalos, sendo 10 um número par).

---
### 2.4 Exemplo

$$
\frac{1}{4 + \sin(20x)}
$$

# Código em Python
~~~python

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def simpson_composta(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("O número de subintervalos (n) deve ser par para a regra de Simpson composta.")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    soma_impar = np.sum(y[1:n:2])   # índices 1, 3, 5, ..., n-1
    soma_par   = np.sum(y[2:n-1:2])   # índices 2, 4, 6, ..., n-2

    return h / 3 * (y[0] + y[n] + 4 * soma_impar + 2 * soma_par)


def plot_simpson_composta_interactive(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("O número de subintervalos (n) deve ser par para a regra de Simpson composta.")

    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)

    # Pontos densos para a curva real da função
    xx = np.linspace(a, b, 1000)
    yy = f(xx)

    # Criação da figura com template dark
    fig = go.Figure()

    # Traço da função real (curva contínua)
    fig.add_trace(go.Scatter(x=xx, y=yy,
                             mode='lines',
                             line=dict(color='blue', width=2),
                             showlegend=False))

    # Traço dos pontos de aproximação e conexão entre eles
    fig.add_trace(go.Scatter(x=x, y=y,
                             mode='lines+markers',
                             marker=dict(color='red', size=8),
                             line=dict(color='red', width=1),
                             showlegend=False))

    # Aproximação por parábolas para cada par de subintervalos
    for i in range(0, n, 2):
        # Gerando pontos para a parábola entre x[i] e x[i+2]
        x_parabola = np.linspace(x[i], x[i+2], 100)
        # Ajuste quadrático pelos três pontos
        coef = np.polyfit([x[i], x[i+1], x[i+2]], [y[i], y[i+1], y[i+2]], 2)
        y_parabola = np.polyval(coef, x_parabola)

        # Criação do polígono para preencher a área entre a parábola e o eixo x
        # Concatena os pontos da curva com os pontos da linha base (y=0) em ordem reversa
        polygon_x = np.concatenate([x_parabola, x_parabola[::-1]]).tolist()
        polygon_y = np.concatenate([y_parabola, np.zeros_like(y_parabola)]).tolist()

        fig.add_trace(go.Scatter(x=polygon_x, y=polygon_y,
                                 fill='toself',
                                 mode='none',
                                 fillcolor='rgba(255,165,0,0.4)',  # laranja com transparência
                                 line=dict(color='black'),
                                 showlegend=False))

    # Configuração final do layout
    fig.update_layout(template='plotly_dark',
                      title_text='Aproximação da Integral pela Regra de Simpson Composta',
                      xaxis_title='x',
                      yaxis_title='f(x)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show()

f = lambda x:  1/(4 + np.sin(20*x))

r_simpson = simpson_composta(f, 0, 3, 30)
plot_simpson_composta_interactive(f, 0, 3, 30)

~~~

<grafico>

---
## 3. Comparação e Considerações sobre o Erro

### 3.1 Erro na Regra Trapezoidal

- O erro de truncamento na regra trapezoidal simples é da ordem de $O((b-a)^3 f''(\xi))$ para algum $\xi \in [a,b]$.
- Na versão composta, o erro total é da ordem de $O(h^2)$, onde $h$ é o tamanho de cada subintervalo.

### 3.2 Erro na Regra de Simpson

- O erro da regra de Simpson simples é da ordem de $O((b-a)^5 f^{(4)}(\xi))$.
- A regra composta de Simpson apresenta erro da ordem de $O(h^4)$, sendo geralmente muito mais precisa que a regra trapezoidal, especialmente para funções com boa suavidade.

### 3.3 Intuição Comparativa

- **Regra Trapezoidal:** Aproxima a função por linhas retas; é simples e funciona razoavelmente bem para funções quase lineares.
- **Regra de Simpson:** Aproxima a função por parábolas; como as parábolas conseguem capturar a curvatura da função, o método geralmente apresenta uma precisão superior com o mesmo número de subintervalos.
- **Versões Compostas:** Dividir o intervalo em subintervalos menores permite capturar melhor as variações da função, reduzindo o erro global da aproximação.

---

## 4. Exemplo Prático

Considere a integração da função $f(x) = \sin(x)$ no intervalo $[0, \pi]$. Ao aplicar tanto a regra trapezoidal composta quanto a regra de Simpson composta, é possível comparar os resultados obtidos com o valor exato da integral, que é $2$. Em geral, a regra de Simpson composta fornecerá uma aproximação muito próxima do valor exato.

---

## 5. Conclusão

Os métodos de integração numérica – tanto a regra trapezoidal quanto a de Simpson – oferecem formas práticas de aproximar integrais definidas. Enquanto a regra trapezoidal é mais simples e intuitiva, a regra de Simpson geralmente proporciona maior precisão, especialmente quando utilizada em sua forma composta. A escolha do método depende da função, do intervalo de integração e da precisão desejada.
