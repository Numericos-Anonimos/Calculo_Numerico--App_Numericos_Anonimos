# Método de Heron (Newton)
## Fundamentos

O **método de Heron / Newton** é um algoritmo iterativo utilizado para calcular a raiz quadrada de um número. Também conhecido como método da média aritmética ou método de aproximação sucessiva, ele se baseia na ideia de que uma aproximação para $\sqrt{S}$ pode ser melhorada iterativamente, tomando a média entre um palpite e o quociente do número $S$ pelo palpite.

### Pré-condições

- O número $S$, do qual se deseja extrair a raiz quadrada, deve ser **positivo**.
- É necessário um palpite inicial $x_0 > 0$. Um valor próximo da raiz verdadeira acelera a convergência.

---

## Algoritmo

1. **Entrada**:
   - Número $S$ (com $S > 0$).
   - Palpite inicial $x_0$ (com $x_0 > 0$).
   - Tolerância $\varepsilon$ (critério de parada).
   - Número máximo de iterações $N_{\text{max}}$ (opcional).

2. **Passos**:
   - Para $i = 0, 1, 2, \dots, N_{\text{max}}$:
     1. Calcule o próximo palpite:
        $$
        x_{i+1} = \frac{x_i + \frac{S}{x_i}}{2}
        $$
     2. Se $|x_{i+1} - x_i| < \varepsilon$, retorne $x_{i+1}$ como a aproximação da raiz quadrada.
   - Se não convergir em $N_{\text{max}}$ iterações, retorne um erro ou a última aproximação calculada.

---

## Código em Python
~~~python
import plotly.express as px
import numpy as np
import pandas as pd

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
                  title='Método de Heron (Newton)',
                  labels={'x': 'x', 'y': 'f(x)'},
                  template='plotly_dark')
    
    fig.add_scatter(x=df_pontos['x'], y=df_pontos['y'],
                    mode='markers', name='Iterações',
                    marker=dict(color='cyan'))
    
    fig.add_scatter(x=[raiz], y=[f(raiz)], mode='markers', 
                    marker=dict(color='yellow', size=12, symbol='diamond'),
                    name='Ponto Final')
    
    return fig


f = lambda x: x**2 - 2
df = lambda x: 2*x
x0 = 1
raiz, pontos = heron(f, df, x0)
xmin, xmax = 0, 2
fig = plot_heron(f, raiz, pontos, xmin, xmax)
fig.show()
~~~

---

## Vantagens

- **Simplicidade**: Fácil de implementar e compreender.
- **Convergência Rápida**: Apresenta convergência quadrática, ou seja, a precisão aumenta significativamente a cada iteração.
- **Eficiência**: Geralmente, poucas iterações são necessárias para atingir alta precisão.

---

## Desvantagens

- **Dependência do Palpite Inicial**: Um palpite inicial muito distante da raiz verdadeira pode levar a uma convergência mais lenta.
- **Restrição de Aplicação**: Destinado especificamente para o cálculo de raízes quadradas de números positivos.
- **Aplicabilidade Limitada**: Não é aplicável diretamente para resolver outros tipos de equações sem adaptações.

---

## Aplicações

- Cálculo manual e computacional de raízes quadradas.
- Implementações em calculadoras e dispositivos embarcados.
- Métodos numéricos que exigem extração de raízes quadradas em problemas de engenharia, física e matemática.

---

## Número de Iterações

Devido à convergência quadrática, o método de Heron geralmente atinge a precisão desejada em poucas iterações, especialmente se o palpite inicial for adequado. Cada iteração melhora exponencialmente a aproximação, tornando-o um método altamente eficiente para a extração de raízes quadradas.

