# Método das Secantes
## Fundamentos

O **método das secantes** é um algoritmo numérico iterativo para encontrar raízes de funções. Ele utiliza a aproximação da derivada por meio da inclinação da reta secante que passa por dois pontos da função, eliminando a necessidade de calcular a derivada explicitamente, como é feito no método de Newton-Raphson.

---

## Pré-condições

- A função $f(x)$ deve ser **contínua** no intervalo de interesse.
- São necessários **dois palpites iniciais** $x_0$ e $x_1$, que idealmente estejam próximos da raiz.
- A convergência não é garantida se os palpites iniciais forem mal escolhidos.

---

## Algoritmo

1. **Entrada**:
   - Função $f(x)$.
   - Dois palpites iniciais $x_0$ e $x_1$.
   - Tolerância $\varepsilon$ (critério de parada).
   - Número máximo de iterações $N_{\text{max}}$.

2. **Passos**:
   - Para $i = 1, 2, \dots, N_{\text{max}}$:
     1. Calcular o próximo palpite:
        $$
        x_{i+1} = x_i - f(x_i) \cdot \frac{x_i - x_{i-1}}{f(x_i) - f(x_{i-1})}
        $$
     2. Se $|x_{i+1} - x_i| < \varepsilon$ ou $|f(x_{i+1})| < \varepsilon$, retorne $x_{i+1}$ como a raiz.
     3. Atualize os palpites: defina $x_{i-1} \leftarrow x_i$ e $x_i \leftarrow x_{i+1}$.
   - Se não convergir em $N_{\text{max}}$ iterações, retorne um erro ou a melhor aproximação encontrada.

---

## Código em Python
~~~python
import numpy as np
import pandas as pd
import plotly.express as px

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
    
    df_func = pd.DataFrame({'x': x, 'f(x)': y,})
    df_pontos = pd.DataFrame({'x': pontos, 'y': f(pontos), 'Iterações': range(len(pontos))})

    fig = px.line(df_func, x='x', y='f(x)', title='Método das Secantes', 
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

    """    for i in range(len(pontos)-1):
        x_line = [pontos[i], pontos[i+1]]
        y_line = [f(pontos[i]), f(pontos[i+1])]
        fig.add_scatter(x=x_line, y=y_line, mode='lines', 
                        line=dict(color='red', dash='dash'), name='Secantes')
    """
    return fig

x0, x1 = 1, 2
f = lambda x: x**2 - 2
raiz, pontos = secantes(f, x0, x1)
xmin, xmax = 0, 3
fig = plot_secantes(f, raiz, pontos, xmin, xmax)
fig.show()
~~~

<grafico>

---


## Vantagens

- **Não requer derivada**:  
  Não é necessário calcular $f'(x)$, o que pode ser vantajoso para funções complicadas ou quando a derivada é difícil de obter.
- **Implementação Simples**:  
  O método é simples e direto de implementar.

---

## Desvantagens

- **Convergência Menos Rápida**:  
  Geralmente, a convergência é superlinear, mas pode ser mais lenta ou instável se os palpites iniciais não forem bem escolhidos.
- **Sensibilidade aos Palpites Iniciais**:  
  Uma escolha inadequada dos palpites pode levar à divergência ou a uma convergência lenta.

---

## Aplicações

- Encontrar raízes de funções onde o cálculo da derivada é complexo ou inexistente.
- Problemas em que se deseja evitar o custo computacional do cálculo de derivadas.

---

## Diferenças com o Método de Newton

- **Cálculo da Derivada**:
  - *Newton*: Requer o cálculo explícito da derivada $f'(x)$ em cada iteração.
  - *Secantes*: Estima a derivada usando os valores de $f(x)$ em dois palpites consecutivos, eliminando a necessidade de derivação.

- **Convergência**:
  - *Newton*: Possui convergência quadrática, o que pode resultar em uma aproximação mais rápida quando o palpite inicial está próximo da raiz.
  - *Secantes*: Apresenta convergência superlinear, geralmente mais lenta que a do método de Newton e potencialmente menos robusta se os palpites iniciais forem inadequados.

- **Complexidade Computacional**:
  - *Newton*: Pode ser computacionalmente mais custoso se a derivada for difícil de calcular.
  - *Secantes*: Evita o cálculo direto da derivada, o que pode ser vantajoso quando o cálculo analítico é complexo ou indisponível.

Em resumo, embora o método de Newton possa oferecer uma convergência mais rápida devido ao uso explícito da derivada, o método das secantes é uma alternativa prática que dispensa o cálculo da derivada, tornando-o útil em situações onde essa operação é complicada ou inviável.

---