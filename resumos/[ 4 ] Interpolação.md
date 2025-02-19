# Interpolação: 
## Fundamentos

O **polinômio interpolador** é um polinômio que passa exatamente por um conjunto de pontos dados, ou seja, ele é construído de modo que sua curva conecte esses pontos sem erro.  

Dado um conjunto de \( n+1 \) pontos distintos:
$$(x_0, y_0), (x_1, y_1), \dots, (x_n, y_n),$$

o **polinômio interpolador** \( P_n(x) \) é um polinômio de grau no máximo \( n \) que satisfaz:
$$P_n(x_i) = y_i, \quad \text{para } i = 0, 1, 2, \dots, n.$$

Isso significa que o polinômio passa **exatamente** pelos pontos fornecidos.

---

## Pré-condições
- Os valores $x_i$ devem ser **distintos** para evitar divisão por zero na construção do polinômio.
- A função de interpolação requer que tenhamos $n+1$ pontos para construir um polinômio de grau $n$.
- É importante que os pontos estejam ordenados ou pelo menos possam ser identificados de forma consistente (embora não seja estritamente necessário, facilita o processo de construção).

--- 

## Método de Lagrange
O **polinômio interpolador de Lagrange** é dado por:
$$P_n(x) = \sum_{i=0}^{n} y_i \, L_i(x),$$

onde os **polinômios de base de Lagrange** $L_i(x)$ são definidos como:

$$L_i(x) = \prod_{\substack{j=0 \\ j \neq i}}^{n} \frac{x - x_j}{x_i - x_j}.$$

Cada $L_i(x)$ vale **1** em $x_i$ e **0** nos outros pontos $x_j$, garantindo que $P_n(x)$ passe exatamente pelos valores $y_i$ fornecidos.


### Código em Python
```python
import numpy as np
import plotly.graph_objects as go

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
                    title="Polinômio Interpolador",
                    labels={"x": "x", "y": "y"},
                    template="plotly_dark",
                    color_discrete_sequence=["cyan"])

    fig.add_traces(px.line(df_polynomial, x="x", y="y",
                            template="plotly_dark",
                            color_discrete_sequence=["yellow"]).data)

    return fig

x = np.array([1, 2, 3, 4], dtype=float)
y = np.array([2, 3, 5, 4], dtype=float)
coef = lagrange(x, y)

fig = plotar_polinomio(x, y, coef, 0, 5, 100)
fig.show()
```

## Método de Newton
Outra maneira eficiente de construir o polinômio interpolador é pela **forma de Newton**, que utiliza **diferenças divididas**. O polinômio de Newton pode ser escrito como:

$$P_n(x) = a_0 \;+\; a_1 (x - x_0) \;+\; a_2 (x - x_0)(x - x_1) \;+\; \dots \;+\; a_n \prod_{k=0}^{n-1} (x - x_k),$$

onde os coeficientes \( a_i \) são calculados a partir das **diferenças divididas**:

$$a_0 = y_0, \quad a_1 = \frac{y_1 - y_0}{x_1 - x_0}, \quad a_2 = \frac{\frac{y_2 - y_1}{x_2 - x_1} - \frac{y_1 - y_0}{x_1 - x_0}}{x_2 - x_0}, \quad \text{etc.}$$

Essa forma é vantajosa porque permite a **adição de novos pontos** sem recalcular todo o polinômio do zero, aproveitando a estrutura de diferenças divididas já computada.

### Código em Python
```python
import numpy as np
import plotly.graph_objects as go

x = np.array([1, 2, 3, 4], dtype=float)
y = np.array([2, 3, 5, 4], dtype=float)
coef = polinomio_newton(x, y)

fig = plotar_polinomio(x, y, coef, 0, 5, 100)
fig.show()
```


## Aplicação Prática
A interpolação polinomial é amplamente utilizada em áreas como:
- Processamento de Sinais
- Modelagem de Dados
- Gráficos Computacionais

## Conclusão
A interpolação polinomial é uma ferramenta valiosa que permite prever valores dentro de um intervalo de dados conhecidos. O método de Lagrange é mais intuitivo, enquanto o método de Newton é mais prático para situações dinâmicas onde novos pontos podem ser adicionados. Ambos são essenciais no toolkit de qualquer pessoa que trabalhe com dados e modelagem matemática.
