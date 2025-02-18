# Método da Bisseção
## Fundamentos

O **método da bissecção** é um algoritmo numérico para encontrar raízes de funções contínuas. Baseia-se no **Teorema do Valor Intermediário**, que garante que, se uma função contínua $f(x)$ muda de sinal em um intervalo $[a, b]$, então existe pelo menos uma raiz nesse intervalo.

### Pré-condições
- A função $f(x)$ deve ser **contínua** em $[a, b]$.
- Os valores $f(a)$ e $f(b)$ devem ter **sinais opostos** (ou seja, $f(a) \cdot f(b) < 0$).

---

## Algoritmo

1. **Entrada**:
   - Função $f(x)$.
   - Intervalo inicial $[a, b]$.
   - Tolerância $\varepsilon$ (critério de parada).
   - Número máximo de iterações $N_{\text{max}}$.

2. **Passos**:
   - Para $i = 1, 2, \dots, N_{\text{max}}$:
     1. Calcule o ponto médio: $c = \frac{a + b}{2}$.
     2. Se $|f(c)| < \varepsilon$ ou $|b - a| < \varepsilon$, retorne $c$ como raiz.
     3. Se $f(a) \cdot f(c) < 0$, atualize $b = c$.
     4. Caso contrário, atualize $a = c$.
   - Se não convergir em $N_{\text{max}}$, retorne erro.
   - `Comentário dos Universitários`: Esse método vai dividindo o intervalo pela metade até encontrar a raiz. É semelhante ao algoritmo de busca binária em um vetor ordenado.

---

# Código em Python
~~~python
import plotly.express as px
import numpy as np

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


def plotar_bisseccao(f, ini, fim, raiz, pontos, xmin, xmax, n):
    x = np.linspace(xmin, xmax, n)
    y = f(x)

    fig = px.line(x = x, y = y, title = 'Método da Bisseção',
                  labels = {'x': 'x', 'y': 'f(x)'},
                  template='plotly_dark')

    fig.add_scatter(x = pontos, y = [f(p) for p in pontos],
        mode = 'markers+lines', marker = dict(color='cyan', size=8),
        line = dict(color='gray', dash='dot'), name = 'Passos da Bissecção')

    fig.add_scatter(x = [raiz], y = [0], mode = 'markers',
        marker = dict(color='red', size=12), name = f'Raiz Final: {raiz:.9f}')

    fig.add_vline(x=ini, line_dash="dash", line_color="green", annotation_text="Início (a)")
    fig.add_vline(x=fim, line_dash="dash", line_color="green", annotation_text="Fim (b)")

    return fig

f = lambda x: x**2 - 2
raiz, pontos = bisseccao(f, 1.0, 2.0)
fig = plotar_bisseccao(f, 1.0, 2.0, raiz, pontos, 0.0, 3.0, 1000)
fig.show()
~~~

<grafico>

---

## Vantagens

- **Simplicidade**: Fácil implementação.
- **Convergência garantida**: Sempre converge para uma raiz se as pré-condições forem satisfeitas.
- **Robustez**: Não requer derivadas da função.

---

## Desvantagens

- **Convergência lenta**: Taxa de convergência linear (a cada iteração, a tolerância é reduzida pela metade).
- **Restrição**: Exige um intervalo inicial com mudança de sinal.
- **Raiz única**: Identifica apenas uma raiz por intervalo.

---

## Aplicações

- Funções contínuas não diferenciáveis.
- Quando a precisão absoluta não é crítica.
- Como método inicial para refinar intervalos antes de usar métodos mais rápidos (ex.: Newton-Raphson).

---

## Número de Iterações

O número mínimo de iterações $n$ para atingir uma tolerância $\varepsilon$ é aproximado por:
$$
n \geq \log_2\left( \frac{b - a}{\varepsilon} \right)
$$

---
    