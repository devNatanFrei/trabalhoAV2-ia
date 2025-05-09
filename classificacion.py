import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# Funções auxiliares e MLP
# ============================

def carregar_dados(caminho):
    dados = np.loadtxt(caminho, delimiter=",")
    return dados[:, :3], dados[:, 3].astype(int)

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoide(x):
    s = sigmoide(x)
    return s * (1 - s)

def calcular_metricas(y_verdadeiro, y_predito):
    VP = np.sum((y_verdadeiro == 1) & (y_predito == 1))
    VN = np.sum((y_verdadeiro == 0) & (y_predito == 0))
    FP = np.sum((y_verdadeiro == 0) & (y_predito == 1))
    FN = np.sum((y_verdadeiro == 1) & (y_predito == 0))
    acuracia = (VP + VN) / len(y_verdadeiro)
    sensibilidade = VP / (VP + FN + 1e-10)
    especificidade = VN / (VN + FP + 1e-10)
    return acuracia, sensibilidade, especificidade, VP, VN, FP, FN

def plotar_curva_aprendizado(acuracias_treino, acuracias_teste, titulo):
    plt.plot(acuracias_treino, label="Treinamento")
    plt.plot(acuracias_teste, label="Teste")
    plt.title(titulo)
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid(True)
    plt.show()

class MLP:
    def __init__(self, camadas, taxa_aprendizado=0.01, epocas=100):
        self.camadas = camadas
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.pesos = self._inicializar_pesos()

    def _inicializar_pesos(self):
        pesos = {}
        for i in range(len(self.camadas) - 1):
            pesos[f'W{i}'] = np.random.randn(self.camadas[i], self.camadas[i+1]) * 0.1
            pesos[f'b{i}'] = np.zeros(self.camadas[i+1])
        return pesos

    def treinar(self, X, y):
        y = y.reshape(-1, 1)
        for epoca in range(self.epocas):
            A, Z = [X], []
            for i in range(len(self.camadas) - 1):
                z = A[i] @ self.pesos[f'W{i}'] + self.pesos[f'b{i}']
                Z.append(z)
                A.append(sigmoide(z))

            erro = (A[-1] - y) * derivada_sigmoide(Z[-1])
            for i in reversed(range(len(self.camadas) - 1)):
                grad_W = A[i].T @ erro
                grad_b = np.sum(erro, axis=0)
                self.pesos[f'W{i}'] -= self.taxa_aprendizado * grad_W
                self.pesos[f'b{i}'] -= self.taxa_aprendizado * grad_b
                if i > 0:
                    erro = (erro @ self.pesos[f'W{i}'].T) * derivada_sigmoide(Z[i-1])
        return self

    def prever(self, X):
        A = X
        for i in range(len(self.camadas) - 1):
            Z = A @ self.pesos[f'W{i}'] + self.pesos[f'b{i}']
            A = sigmoide(Z)
        return (A >= 0.5).astype(int).flatten()

def simulacao_monte_carlo(X, y, camadas, R=250, epocas=100):
    acuracias, sensibilidades, especificidades = [], [], []

    # Inicialização segura com estrutura completa
    melhor = {'acuracia': 0, 'sens': 0, 'esp': 0, 'confusao': (0, 0, 0, 0), 'modelo': None, 'X': None, 'y': None}
    pior = {'acuracia': 1, 'sens': 0, 'esp': 0, 'confusao': (0, 0, 0, 0), 'modelo': None, 'X': None, 'y': None}

    for r in range(R):
        indices = np.random.permutation(len(X))
        tamanho_treino = int(0.8 * len(X))
        idx_treino, idx_teste = indices[:tamanho_treino], indices[tamanho_treino:]
        X_treino, X_teste = X[idx_treino], X[idx_teste]
        y_treino, y_teste = y[idx_treino], y[idx_teste]

        mlp = MLP(camadas, epocas=epocas)
        mlp.treinar(X_treino, y_treino)
        y_predito = mlp.prever(X_teste)

        acuracia, sensibilidade, especificidade, VP, VN, FP, FN = calcular_metricas(y_teste, y_predito)
        acuracias.append(acuracia)
        sensibilidades.append(sensibilidade)
        especificidades.append(especificidade)

        if acuracia > melhor['acuracia']:
            melhor.update({'acuracia': acuracia, 'sens': sensibilidade, 'esp': especificidade, 
                           'confusao': (VP, FN, FP, VN), 'modelo': mlp, 'X': X_teste, 'y': y_teste})
        if acuracia < pior['acuracia']:
            pior.update({'acuracia': acuracia, 'sens': sensibilidade, 'esp': especificidade, 
                         'confusao': (VP, FN, FP, VN), 'modelo': mlp, 'X': X_teste, 'y': y_teste})
    return acuracias, sensibilidades, especificidades, melhor, pior

def plotar_matriz_confusao(confusao, titulo):
    VP, FN, FP, VN = confusao
    matriz = np.array([[VP, FN], [FP, VN]])
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Predito 1", "Predito 0"], 
                yticklabels=["Real 1", "Real 0"])
    plt.title(titulo)
    plt.show()

def plotar_box_violin(dados, rotulos, titulo, tipo="box"):
    plt.figure(figsize=(8, 6))
    if tipo == "box":
        sns.boxplot(data=dados)
    else:
        sns.violinplot(data=dados)
    plt.xticks(ticks=np.arange(len(rotulos)), labels=rotulos)
    plt.title(titulo)
    plt.grid(True)
    plt.show()

# ========================
# Execução dos modelos
# ========================
X, y = carregar_dados("Spiral3d.csv")

topologias = {
    "Subdimensionada": [3, 2, 1],
    "Adequada": [3, 10, 1],
    "Superdimensionada": [3, 50, 30, 1]
}

resultados = {}
for nome, camadas in topologias.items():
    print(f"\nRodando Monte Carlo - {nome}")
    acuracias, sensibilidades, especificidades, melhor, pior = simulacao_monte_carlo(X, y, camadas)
    resultados[nome] = {
        "acuracia": acuracias, "sens": sensibilidades, "esp": especificidades,
        "melhor": melhor, "pior": pior
    }

# ===========================
# Gráficos e análises finais
# ===========================

# 6 - Matrizes de confusão
for nome in resultados:
    print(f"\n{nome} - Melhor")
    if resultados[nome]["melhor"]["modelo"] is not None:
        plotar_matriz_confusao(resultados[nome]["melhor"]['confusao'], f"{nome} - Maior Acurácia")
    print(f"{nome} - Pior")
    if resultados[nome]["pior"]["modelo"] is not None:
        plotar_matriz_confusao(resultados[nome]["pior"]['confusao'], f"{nome} - Menor Acurácia")

# 7 - Tabelas e boxplots
for metrica in ['acuracia', 'sens', 'esp']:
    dados = [resultados[n][metrica] for n in topologias]
    rotulos = list(topologias.keys())
    print(f"\nMétrica: {metrica.upper()}")
    for i, nome in enumerate(rotulos):
        media = np.mean(dados[i])
        desvio = np.std(dados[i])
        maximo = np.max(dados[i])
        minimo = np.min(dados[i])
        print(f"{nome:18} | Média: {media:.4f} | Desvio: {desvio:.4f} | Máx: {maximo:.4f} | Mín: {minimo:.4f}")
    plotar_box_violin(dados, rotulos, f"{metrica.upper()} - Boxplot", "box")
    plotar_box_violin(dados, rotulos, f"{metrica.upper()} - Violin Plot", "violin")
