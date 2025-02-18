# Método de Newton

## Fundamentos

O **método de Newton para sistemas não lineares** é uma extensão do método de Newton-Raphson para encontrar raízes de sistemas de equações não lineares. Ele é usado para resolver sistemas da forma:

$$
F(\mathbf{x}) = 0
$$

onde $F: \mathbb{R}^n \to \mathbb{R}^n$ é uma função vetorial com $n$ equações e $n$ incógnitas. O método utiliza a aproximação linear de $F$ em torno de uma estimativa $\mathbf{x}^{(k)}$, empregando a **matriz Jacobiana** para obter a correção de Newton.

---

## Jacobiana

Para um sistema de $n$ equações, $F(\mathbf{x}) = (f_1(\mathbf{x}), f_2(\mathbf{x}), \ldots, f_n(\mathbf{x}))^T$, a **matriz Jacobiana** é definida como:

$$
J(\mathbf{x}) \;=\;
\begin{bmatrix}
\dfrac{\partial f_1}{\partial x_1} & \dfrac{\partial f_1}{\partial x_2} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\
\dfrac{\partial f_2}{\partial x_1} & \dfrac{\partial f_2}{\partial x_2} & \cdots & \dfrac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\dfrac{\partial f_n}{\partial x_1} & \dfrac{\partial f_n}{\partial x_2} & \cdots & \dfrac{\partial f_n}{\partial x_n}
\end{bmatrix}.
$$

Ela generaliza o conceito de derivada para funções de várias variáveis. No método de Newton para sistemas, a Jacobiana avaliada em $\mathbf{x}^{(k)}$ fornece as derivadas parciais necessárias para aproximar o comportamento local de $F(\mathbf{x})$ em torno de $\mathbf{x}^{(k)}$. A cada iteração, resolve-se o sistema linear

$$
J(\mathbf{x}^{(k)})\, \mathbf{d}^{(k)} = -\,F(\mathbf{x}^{(k)}),
$$

para determinar o **passo de Newton** $\mathbf{d}^{(k)}$ que será usado na atualização da solução aproximada.

---

## Pré-condições

- A função $F(\mathbf{x})$ deve ser **diferenciável** e contínua no domínio de interesse.
- O palpite inicial $\mathbf{x}^{(0)}$ deve ser suficientemente próximo da solução.
- A matriz Jacobiana $J(\mathbf{x})$ não deve ser singular nas iterações.

---

## Algoritmo

1. **Entrada**:
   - Função vetorial $F(\mathbf{x})$.
   - Palpite inicial $\mathbf{x}^{(0)}$.
   - Tolerância $\varepsilon$ (critério de parada).
   - Número máximo de iterações $N_{\text{max}}$.

2. **Passos**:
   - Para $k = 0, 1, 2, \dots, N_{\text{max}}$:
     1. Calcular a matriz Jacobiana $J(\mathbf{x}^{(k)})$.
     2. Resolver o sistema linear:
        $$
        J(\mathbf{x}^{(k)})\, \mathbf{d}^{(k)} = -F(\mathbf{x}^{(k)})
        $$
     3. Atualizar a aproximação:
        $$
        \mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \mathbf{d}^{(k)}
        $$
     4. Se $\|\mathbf{d}^{(k)}\| < \varepsilon$ ou $\|F(\mathbf{x}^{(k+1)})\| < \varepsilon$, retorne $\mathbf{x}^{(k+1)}$ como a solução.
   - Se não convergir em $N_{\text{max}}$ iterações, retorne um erro ou a melhor aproximação encontrada.

---

## Exemplo:

Intersecção entre parábola e elipse

$
y = x^2 + 1
$

$
{x^2} + \frac{y^2}{4} = 1
$

~~~python

import numpy as np
import pandas as pd
import plotly.express as px

def newton(Func, JFunc, x0, TOL, N):
    x = np.copy(x0).astype('double')
    k = 0
    x_vals = [x[0]]
    y_vals = [x[1]]

    while k < N:
        k += 1
        delta = np.linalg.solve(JFunc(x), -Func(x))

        x = x + delta

        x_vals.append(x[0])
        y_vals.append(x[1])

        if np.linalg.norm(delta, np.inf) < TOL:
            break
    return x, x_vals, y_vals

def plot_newton_intersecao(Func, JFunc, x0, TOL, N):

    # Executa o método de Newton
    final_x, newton_x, newton_y = newton(Func, JFunc, x0, TOL, N)
    
    # Cria DataFrame com os pontos de iteração do método de Newton
    df_newton = pd.DataFrame({
        "x": newton_x,
        "y": newton_y,
        "Iteração": list(range(len(newton_x)))
    })
    
    # Define a parábola: y = x² + 1 para x entre -2 e 2
    x_par = np.linspace(-2, 2, 400)
    y_par = x_par**2 + 1
    df_parabola = pd.DataFrame({"x": x_par, "y": y_par})
    
    # Define a elipse: x² + y²/4 = 1, resolvendo para y:
    # y = ±2*sqrt(1 - x²) para x entre -1 e 1
    x_ell = np.linspace(-1, 1, 300)
    y_ell_upper = 2 * np.sqrt(1 - x_ell**2)
    y_ell_lower = -2 * np.sqrt(1 - x_ell**2)
    df_ellipse_upper = pd.DataFrame({"x": x_ell, "y": y_ell_upper})
    df_ellipse_lower = pd.DataFrame({"x": x_ell, "y": y_ell_lower})
    
    # Cria gráfico de dispersão para os pontos do método de Newton
    fig = px.scatter(df_newton, x="x", y="y", text="Iteração",
                     title="Método de Newton: Intersecção entre Parábola e Elipse",
                     labels={"x": "x", "y": "y"},
                     template="plotly_dark",
                     color_discrete_sequence=["cyan"])
    
    # Adiciona traço de linha conectando os pontos do método de Newton
    fig.add_traces(px.line(df_newton, x="x", y="y",
                           template="plotly_dark",
                           color_discrete_sequence=["yellow"]).data)
    
    # Adiciona traço para a parábola
    fig.add_traces(px.line(df_parabola, x="x", y="y",
                           template="plotly_dark",
                           color_discrete_sequence=["magenta"],
                           labels={"x": "x", "y": "y"},
                           title="Parábola: y = x² + 1").data)
    
    # Adiciona traços para a parte superior da elipse
    fig.add_traces(px.line(df_ellipse_upper, x="x", y="y",
                           template="plotly_dark",
                           color_discrete_sequence=["orange"],
                           labels={"x": "x", "y": "y"},
                           title="Elipse: x² + y²/4 = 1").data)
    
    # Adiciona traços para a parte inferior da elipse
    fig.add_traces(px.line(df_ellipse_lower, x="x", y="y",
                           template="plotly_dark",
                           color_discrete_sequence=["orange"]).data)
    
    fig.update_traces(textposition='top center')
    fig.show()

def F(x):
    f1 = x[1] - (x[0]**2 + 1)
    f2 = x[0]**2 + (x[1]**2)/4 - 1
    return np.array([f1, f2])

def JF(x):
    return np.array([[-2*x[0],    1],
                     [ 2*x[0], x[1]/2]])

x0 = np.array([2.0, 4.0])  
TOL = 1e-5
N = 100

plot_newton_intersecao(F, JF, x0, TOL, N)

~~~

<grafico>

---

---
## Vantagens e desvantagens

### Vantagens:
- **Convergência Rápida**: O método converge de forma quadrática quando está perto da solução.
- **Alta Precisão**: Fornece soluções precisas quando converge.
- **Versatilidade**: Pode ser aplicado a sistemas não lineares em várias dimensões.
- **Fácil Implementação**: O algoritmo é simples, exigindo apenas a matriz Jacobiana e a função residual.

### Desvantagens:
- **Dependência de Aproximação Inicial**: Requer uma boa aproximação inicial para garantir a convergência.
- **Cálculo da Jacobiana**: Pode ser computacionalmente caro, especialmente para sistemas grandes.
- **Problemas de Convergência**: Pode falhar em sistemas com funções não suaves, pontos de inflexão ou quando a Jacobiana é mal condicionada.
- **Divergência**: Se a escolha inicial ou as condições do sistema não forem adequadas, o método pode divergir.

---

## Conclusão

O método de Newton para sistemas não lineares é uma ferramenta poderosa para resolver equações complexas, destacando-se pela convergência rápida (quadrática) e alta precisão quando a aproximação inicial está próxima da solução. Sua versatilidade permite a aplicação em sistemas de várias dimensões e sua implementação é relativamente simples, necessitando apenas da função residual e da matriz Jacobiana.

Por outro lado, o desempenho do método depende fortemente de uma boa aproximação inicial, e o cálculo da matriz Jacobiana pode se tornar computacionalmente oneroso em sistemas grandes. Além disso, se o sistema envolver funções não suaves, pontos de inflexão ou uma Jacobiana mal condicionada, o método pode enfrentar problemas de convergência ou até mesmo divergir.

Em resumo, embora o método de Newton seja altamente eficaz quando utilizado nas condições adequadas, é crucial avaliar cuidadosamente a escolha do palpite inicial e as características do sistema para garantir seu sucesso.

