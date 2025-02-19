# Método de Jacobi e Gauss-Seidel
## Fundamentos

Os métodos iterativos de Gauss são técnicas numéricas para resolver sistemas lineares da forma: 
$$Ax = b,$$

onde \(A\) é a matriz de coeficientes, \(x\) o vetor incógnita e \(b\) o vetor dos termos independentes. Entre os métodos mais comuns estão o **Jacobi** e o **Gauss-Seidel**, que geram sequências de aproximações para \(x\) até que se atinja a convergência.

---

## Método de Jacobi
### Algoritmo

Para um sistema \(Ax = b\), decompomos \(A\) na soma de uma matriz diagonal \(D\) e uma matriz restante \(R\), isto é, \(A = D + R\). A iteração do método de Jacobi é dada por:

$$x^{(k+1)} = D^{-1} \left(b - R x^{(k)}\right).$$

Isso resulta na seguinte fórmula para cada componente \(i\) do vetor:

$$x_i^{(k+1)} = \frac{1}{a_{ii}} \left(b_i - \sum_{j \neq i} a_{ij} x_j^{(k)}\right).$$

### Código em Python    
Considere o sistema:
$$\begin{cases} 4x_1 + x_2 = 9, \\ 2x_1 + 3x_2 = 8. \end{cases}$$

~~~python
import numpy as np
import pandas as pd
import plotly.express as px

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

def plot_jacobi_gauss_seidel(jacobi=None, gauss_seidel=None):
    nome = "Comparação entre Jacobi e Gauss-Seidel" if jacobi and gauss_seidel else ("Método de Jacobi" if jacobi else "Método de Gauss-Seidel")
    fig = px.line(template="plotly_dark", title=nome)
    
    if jacobi:
        df_jacobi = criar_dataframe(jacobi, "Jacobi")
        fig.add_traces(px.line(df_jacobi, x="Iteração", y="Valor",
                               color="Variável",
                               markers=True).data)
    if gauss_seidel:
        df_gauss_seidel = criar_dataframe(gauss_seidel, "Gauss-Seidel")
        fig.add_traces(px.line(df_gauss_seidel, x="Iteração", y="Valor",
                               color="Variável",
                               markers=True, line_dash_sequence=['dash']).data)
    
    return fig

A = np.array([[4.0, 2.0],
              [1.0, 3.0]])
b = np.array([9.0, 8.0])
x0 = np.array([1.0, 1.0])
x_jacobi, iter_jacobi = jacobi_solver(A, b, x0, tol=1e-6, max_iter=25)
fig = plot_jacobi(iter_jacobi)
fig.show()
~~~


---

## Método de Gauss-Seidel
### Algoritmo

O método de Gauss-Seidel também visa resolver \(Ax = b\), mas, em cada iteração, utiliza os valores já atualizados dos componentes de \(x\). A fórmula para a atualização de cada componente é:

$$x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j=1}^{i-1} a_{ij} x_j^{(k+1)} - \sum_{j=i+1}^{n} a_{ij} x_j^{(k)} \right).$$

### Código em Python
~~~python
import numpy as np
import pandas as pd
import plotly.express as px

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

A = np.array([[4.0, 2.0],
              [1.0, 3.0]])
b = np.array([9.0, 8.0])
x0 = np.array([1.0, 1.0])

x_gauss_seidel, iter_gauss_seidel = gauss_seidel_solver(A, b, x0, tol=1e-6, max_iter=25)
x_jacobi, iter_jacobi = jacobi_solver(A, b, x0, tol=1e-6, max_iter=25)
fig = plot_jacobi_gauss_seidel(iter_jacobi, iter_gauss_seidel)
fig.show()
~~~



---

## Diferenças, Prós e Contras

### Diferenças

- **Jacobi**:  
  Atualiza todos os componentes simultaneamente utilizando os valores da iteração anterior. Cada $(x_i^{(k+1)})$ depende somente dos valores $(x_j^{(k)})$ para $(j \neq i)$.

- **Gauss-Seidel**:  
  Atualiza os componentes sequencialmente, usando imediatamente os valores já calculados na mesma iteração para atualizar os demais componentes.

### Prós e Contras

**Método de Jacobi:**

- **Prós:**
  - Fácil de implementar.
  - Fácil de paralelizar, pois as atualizações de cada componente são independentes.
  
- **Contras:**
  - Geralmente, apresenta convergência mais lenta, pois não utiliza os valores atualizados imediatamente.

**Método de Gauss-Seidel:**

- **Prós:**
  - Convergência geralmente mais rápida, pois utiliza os valores mais recentes na iteração.
  
- **Contras:**
  - Menos adequado para paralelização, já que as atualizações são sequenciais.
  - A convergência pode não ocorrer se a matriz \(A\) não for estritamente diagonal dominante ou não satisfizer outras condições de convergência.

---

## Considerações Finais

Ambos os métodos requerem condições para a garantia de convergência, como a matriz \(A\) ser estritamente diagonal dominante ou simétrica e definida positiva. A escolha entre o método de Jacobi e o de Gauss-Seidel depende do problema específico e da importância de fatores como facilidade de implementação e possibilidade de paralelização.
