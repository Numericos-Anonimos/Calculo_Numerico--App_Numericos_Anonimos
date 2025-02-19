## Métodos de Runge-Kutta

Os métodos de Runge-Kutta geram uma sequência de valores aproximados para $ y(t) $ a partir de uma condição inicial $ y(t_0) = y_0 $. A vantagem desses métodos sobre o método de Euler reside na utilização de informações adicionais sobre $ f(t, y) $ em pontos intermediários, o que resulta em uma aproximação mais precisa.

---

## Fundamentos

O **método de Runge-Kutta** é uma família de métodos numéricos para resolver equações diferenciais ordinárias (EDOs) com maior precisão do que métodos simples, como o de Euler. Em particular, o método de Runge-Kutta de 4ª ordem (RK4) é amplamente utilizado devido ao seu bom equilíbrio entre precisão e eficiência computacional.

Consideramos uma EDO da forma:

$$
\frac{dy}{dt} = f(t, y)
$$

onde $ y = y(t) $ é a função desconhecida e $ f(t, y) $ descreve a taxa de variação de $ y $ em relação ao tempo $ t $.

---

## Método de Runge-Kutta de 4ª Ordem (RK4)

No método RK4, o próximo valor $ y_{n+1} $ é calculado a partir do valor atual $ y_n $ utilizando quatro estimativas intermediárias $ k_1 $, $ k_2 $, $ k_3 $ e $ k_4 $:

$$
k_1 = h \cdot f(t_n, y_n)
$$

$$
k_2 = h \cdot f\left(t_n + \frac{h}{2}, \, y_n + \frac{k_1}{2}\right)
$$

$$
k_3 = h \cdot f\left(t_n + \frac{h}{2}, \, y_n + \frac{k_2}{2}\right)
$$

$$
k_4 = h \cdot f(t_n + h, \, y_n + k_3)
$$

A atualização é realizada pela fórmula:

$$
y_{n+1} = y_n + \frac{1}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)
$$

---

## Método de Runge-Kutta de 2ª Ordem (RK2)

O método RK2 é uma alternativa mais simples que também melhora a precisão em relação ao método de Euler. Existem variantes do RK2; duas das mais comuns são:

### Variante do Ponto Médio

Utiliza uma estimativa no meio do intervalo:

$$
k_1 = h \cdot f(t_n, y_n)
$$

$$
k_2 = h \cdot f\left(t_n + \frac{h}{2}, \, y_n + \frac{k_1}{2}\right)
$$

$$
y_{n+1} = y_n + k_2
$$


---

## Passos dos Métodos RK4 e RK2

1. **Inicialização**:  
   Defina a condição inicial $ y(t_0) = y_0 $.

2. **Discretização do Tempo**:  
   Divida o intervalo de tempo em passos de tamanho $ h $, de forma que $ t_n = t_0 + n \cdot h $.

3. **Cálculo dos Coeficientes Intermediários**:  
   - Para RK4: Calcule $ k_1 $, $ k_2 $, $ k_3 $ e $ k_4 $ conforme definido.
   - Para RK2: Calcule os coeficientes conforme a variante escolhida (Ponto Médio ou Heun).

4. **Atualização da Solução**:  
   Utilize a fórmula de atualização para obter $ y_{n+1} $.

5. **Repetição**:  
   Repita o processo até alcançar o tempo final desejado.

---

## Precisão dos Métodos RK

- **RK4**: Possui erro local de ordem $ h^5 $ e erro global de ordem $ h^4 $, proporcionando uma precisão significativamente maior.
- **RK2**: Embora menos preciso que o RK4, oferece erro local de ordem $ h^3 $ e erro global de ordem $ h^2 $, sendo uma opção intermediária entre Euler e RK4.

---

## Conclusão

Os métodos de Runge-Kutta são fundamentais na resolução numérica de EDOs. O RK4 é ideal quando a precisão é crucial, enquanto o RK2 oferece uma solução mais simples e rápida com uma precisão intermediária. A escolha entre RK2 e RK4 depende dos requisitos de precisão do problema e dos recursos computacionais disponíveis.

---

## Exemplo

A equação diferencial a ser resolvida é:

$$
\frac{dy}{dt} = -2y
$$

Onde:
- $ y(t)$ é a solução que queremos encontrar.
- $ t $ é a variável independente (tempo).

## Parâmetros do Problema:
- **Condição Inicial**: $ y(0) = 1 $
- **Tempo Inicial**: $ t_0 = 0 $
- **Tempo Final**: $ t_{\text{fim}} = 5 $
- **Passo de Tempo**: $ \Delta t = 0.1 $

# Código em Python
~~~python

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    # Atualizando o layout para fundo preto e aparência "plotly_dark"
    fig.update_layout(
        title="Comparação dos Métodos de Runge-Kutta (RK2 vs RK4)",
        xaxis_title="t (tempo)",
        yaxis_title="y",
        template="plotly_dark",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        legend=dict(x=0.02, y=0.98)
    )
    
    # Atualizando os títulos dos subplots
    fig.update_xaxes(title_text="t (tempo)", row=2, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_yaxes(title_text="y", row=2, col=1)
    
    fig.show()

t0 = 0.0
t_end = 2.0
y0 = 1.0
dt = 0.1

plot_runge_kutta(f, y0, t0, t_end, dt)

~~~

---

## Vantagens e desvantagens

### Vantagens:
- **Alta Precisão**: Oferece maior precisão, especialmente a versão de 4ª ordem (RK4).
- **Menor Erro de Aproximação**: Menor erro de truncamento comparado ao Método de Euler.
- **Versatilidade**: Adequado para uma ampla variedade de problemas, incluindo sistemas não lineares.
- **Boa Estabilidade**: Funciona bem em sistemas com boa estabilidade.
- **Sem Necessidade de Jacobiana**: Não requer cálculos extras como a matriz Jacobiana.

### Desvantagens:
- **Custo Computacional Maior**: Requer mais cálculos por iteração, tornando-o mais caro computacionalmente.
- **Menos Eficiente para Sistemas Simples**: Pode ser excessivo para problemas simples.
- **Necessidade de Passos Menores**: Para alta precisão em sistemas rígidos, pode exigir passos muito pequenos.
- **Problemas de Estabilidade em Sistemas Rígidos**: Pode não ser ideal para sistemas extremamente rígidos.
  
---

## Conclusão

O método de Runge-Kutta de 4ª ordem (RK4) é amplamente utilizado na resolução numérica de EDOs devido à sua alta precisão e robustez. Embora seja mais complexo e exija mais cálculos por passo do que o método de Euler, a utilização de múltiplos coeficientes intermediários permite uma melhor aproximação da solução, tornando-o ideal para problemas onde a precisão é crucial e o custo computacional é aceitável.


