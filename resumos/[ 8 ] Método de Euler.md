# Ideia Básica do Método de Euler

O método de Euler baseia-se na **aproximação da derivada** por uma diferença finita. Se \( y(t) \) é diferenciável, podemos aproximar sua derivada em um ponto \( t_n \) por:

$$
\frac{dy}{dt}(t_n) \approx \frac{y(t_{n+1}) - y(t_n)}{h},
$$

onde \( h \) é o tamanho do passo. Assim, a partir do valor inicial \( y(t_0) = y_0 \), o método estima \( y(t_{n+1}) \) usando a fórmula:

$$
y_{n+1} = y_n + h \cdot f(t_n, y_n).
$$

---

## Fundamentos

O **método de Euler** é um dos métodos mais simples e antigos para resolver equações diferenciais ordinárias (EDOs) numericamente. Ele é utilizado para aproximar a solução de uma EDO da forma:

$$
\frac{dy}{dt} = f(t, y)
$$

onde \( y = y(t) \) é a função que desejamos determinar e \( f(t, y) \) descreve a taxa de variação de \( y \) em relação ao tempo \( t \).

---

## Passos do Método de Euler

1. **Condição Inicial**:  
   Defina \( y(t_0) = y_0 \), onde \( t_0 \) é o tempo inicial e \( y_0 \) o valor conhecido de \( y \).

2. **Discretização do Tempo**:  
   Divida o intervalo de tempo em pequenos passos \( h \), de forma que \( t_n = t_0 + n \cdot h \).

3. **Iteração de Euler**:  
   Utilize a fórmula recursiva para calcular os valores de \( y \) em cada ponto:
   
   $$
   y_{n+1} = y_n + h \cdot f(t_n, y_n)
   $$

4. **Repetição**:  
   Repita o processo até que o tempo atinja o valor final desejado.

---

## Exemplo Prático

Considere a EDO:

$$
\frac{dy}{dt} = -2y, \quad y(0) = 1.
$$

Utilizando o método de Euler com um passo \( h = 0.1 \):

1. **Condição Inicial**:  
   \( y_0 = 1 \).

2. **Para \( t_1 = 0.1 \)**:

   $$
   y_1 = y_0 + 0.1 \cdot (-2 \cdot y_0) = 1 + 0.1 \cdot (-2) = 1 - 0.2 = 0.8.
   $$

3. **Para \( t_2 = 0.2 \)**:

   $$
   y_2 = y_1 + 0.1 \cdot (-2 \cdot y_1) = 0.8 + 0.1 \cdot (-1.6) = 0.8 - 0.16 = 0.64.
   $$

Este processo é repetido para os demais valores de \( t \).

---

## Precisão e Limitações

O erro do método de Euler é proporcional ao tamanho do passo \( h \); passos maiores geram erros maiores. Embora seja fácil de implementar, o método de Euler pode não ser adequado para problemas que exigem alta precisão ou para EDOs com comportamentos complexos. Para tais casos, métodos mais sofisticados, como o método de Runge-Kutta, podem ser mais apropriados.

---

## Conclusão

O método de Euler é uma excelente ferramenta introdutória para a resolução numérica de EDOs, permitindo compreender o conceito de aproximação de derivadas por diferenças finitas. Sua simplicidade facilita a implementação e a visualização da evolução de uma solução ao longo do tempo. Contudo, devido à sua dependência do tamanho do passo, pode apresentar limitações em termos de precisão, sendo mais adequado para problemas onde uma solução aproximada é aceitável.

---

## Exemplo

 Calcule y(0.4) usando o Método de Euler com h=0.01 para o PVI:

$$
\begin{cases}
y' = y \\\\
y(0) = 1
\end{cases}
$$

# CCódigo em Python
~~~python

import numpy as np
import plotly.graph_objects as go

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
    
    # Configurar layout com fundo preto e template "plotly_dark"
    fig.update_layout(
        title='Solução da EDO pelo Método de Euler',
        xaxis_title='x',
        yaxis_title='y',
        template='plotly_dark',
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white')
    )
    
    fig.show()


def f(x, y):
    return -2 * y

x0 = 0.0
y0 = 1.0
xf = 2.0
h = 0.1

# Calcular a solução numérica usando o método de Euler
xs, ys = euler(f, x0, y0, xf, h)

~~~

<gráfico>

---

## Vantagens e desvantagens
### Vantagens:
- **Simples de Implementar**: Fácil de entender e implementar.
- **Baixo Custo Computacional**: Requer poucas operações por iteração.
- **Rápido para Problemas Simples**: Funciona bem para problemas simples sem alta precisão.
- **Intuitivo**: Baseado em uma fórmula simples e conceitualmente clara.

### Desvantagens:
- **Baixa Precisão**: Apresenta erro significativo, especialmente com passos grandes.
- **Instabilidade para Passos Grandes**: Pode se tornar instável em equações com comportamento oscilatório.
- **Erro Acumulado**: O erro cresce ao longo das iterações.
- **Não Adequado para Sistemas Rígidos**: Ineficiente para sistemas com variação rápida de soluções.
- **Necessidade de Passos Menores**: Para melhorar a precisão, requer passos pequenos, aumentando o custo computacional.

---

### Conclusão:
O Método de Euler é simples e eficiente para problemas simples, mas apresenta limitações em precisão e estabilidade. Métodos mais avançados são recomendados para sistemas mais complexos.
