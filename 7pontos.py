import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def carregar_dados(path='coluna_vertebral.csv'):
    dados = np.genfromtxt(path, delimiter=',', dtype=str)
    X = dados[1:, :-1].astype(float).T
    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    X = np.vstack([X, np.ones((1, X.shape[1]))])  
    labels = dados[1:, -1]
    Y = np.zeros((3, len(labels)))
    for i, rot in enumerate(labels):
        if rot == 'NO': Y[:, i] = [1, -1, -1]
        elif rot == 'DH': Y[:, i] = [-1, 1, -1]
        elif rot == 'SL': Y[:, i] = [-1, -1, 1]
    return X, Y


def perceptron_simples(X, Y, epocas=600, eta=0.01):
    W = np.random.randn(Y.shape[0], X.shape[0])
    for _ in range(epocas):
        for i in range(X.shape[1]):
            xi = X[:, i:i+1]
            yi = Y[:, i:i+1]
            y_pred = np.sign(W @ xi)
            erro = yi - y_pred
            W += eta * erro @ xi.T
    return W

def adaline(X, Y, epocas=600, eta=0.001):
    W = np.random.randn(Y.shape[0], X.shape[0])
    for _ in range(epocas):
        Y_pred = W @ X
        erro = Y - Y_pred
        W += eta * erro @ X.T / X.shape[1]
    return W

def mlp(X, Y, h=10, epocas=600, eta=0.01):
    p, N = X.shape
    c = Y.shape[0]
    W1 = np.random.randn(h, p)
    b1 = np.random.randn(h, 1)
    W2 = np.random.randn(c, h)
    b2 = np.random.randn(c, 1)
    for _ in range(epocas):
        Z1 = np.tanh(W1 @ X + b1)
        Y_pred = W2 @ Z1 + b2
        erro = Y_pred - Y
        dW2 = erro @ Z1.T / N
        db2 = np.mean(erro, axis=1, keepdims=True)
        dZ1 = (1 - Z1**2) * (W2.T @ erro)
        dW1 = dZ1 @ X.T / N
        db1 = np.mean(dZ1, axis=1, keepdims=True)
        W2 -= eta * dW2
        b2 -= eta * db2
        W1 -= eta * dW1
        b1 -= eta * db1
    return W1, b1, W2, b2


def pred_perceptron(W, X): return np.sign(W @ X)
def pred_adaline(W, X): return np.sign(W @ X)
def pred_mlp(W1, b1, W2, b2, X): return np.sign(W2 @ np.tanh(W1 @ X + b1) + b2)


def calcula_metricas(y_true, y_pred):
    acertos = np.all(y_true == y_pred, axis=0)
    acuracia = np.mean(acertos)
    sens, esp = [], []
    for i in range(y_true.shape[0]):
        TP = np.sum((y_true[i] == 1) & (y_pred[i] == 1))
        TN = np.sum((y_true[i] == -1) & (y_pred[i] == -1))
        FP = np.sum((y_true[i] == -1) & (y_pred[i] == 1))
        FN = np.sum((y_true[i] == 1) & (y_pred[i] == -1))
        sens.append(TP / (TP + FN + 1e-10))
        esp.append(TN / (TN + FP + 1e-10))
    return acuracia, np.mean(sens), np.mean(esp)

def split_data(X, Y, proporcao=0.8):
    N = X.shape[1]
    idx = np.random.permutation(N)
    n_train = int(proporcao * N)
    return X[:, idx[:n_train]], Y[:, idx[:n_train]], X[:, idx[n_train:]], Y[:, idx[n_train:]]

def matriz_confusao(y_true, y_pred):
    def decode(y):
        for i in range(y.shape[1]):
            yi = y[:, i]
            if np.array_equal(yi, [1, -1, -1]): yield 0
            elif np.array_equal(yi, [-1, 1, -1]): yield 1
            elif np.array_equal(yi, [-1, -1, 1]): yield 2
    true, pred = list(decode(y_true)), list(decode(y_pred))
    M = np.zeros((3, 3), dtype=int)
    for t, p in zip(true, pred): M[t, p] += 1
    return M


def monte_carlo(X, Y, R=100):
    resultados = {k: [] for k in ['Perceptron', 'ADALINE', 'MLP']}
    confs = []
    for _ in range(R):
        Xt, Yt, Xs, Ys = split_data(X, Y)
        Wp = perceptron_simples(Xt, Yt)
        Wa = adaline(Xt, Yt)
        W1, b1, W2, b2 = mlp(Xt, Yt)

        yp = pred_perceptron(Wp, Xs)
        ya = pred_adaline(Wa, Xs)
        ym = pred_mlp(W1, b1, W2, b2, Xs)

        resultados['Perceptron'].append(calcula_metricas(Ys, yp))
        resultados['ADALINE'].append(calcula_metricas(Ys, ya))
        resultados['MLP'].append(calcula_metricas(Ys, ym))

        confs.append({
            'Perceptron': matriz_confusao(Ys, yp),
            'ADALINE': matriz_confusao(Ys, ya),
            'MLP': matriz_confusao(Ys, ym),
        })
    return resultados, confs

def extrair_acuracias(modelo, resultados):
    return np.array([r[0] for r in resultados[modelo]])

def melhor_pior_idx(modelo, resultados):
    acs = extrair_acuracias(modelo, resultados)
    return np.argmax(acs), np.argmin(acs)

def estatisticas(modelo, resultados):
    acs = extrair_acuracias(modelo, resultados)
    return {
        'media': np.mean(acs),
        'desvio': np.std(acs),
        'max': np.max(acs),
        'min': np.min(acs)
    }

def plot_confusoes(confs, resultados):
    for modelo in resultados:
        best, worst = melhor_pior_idx(modelo, resultados)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.heatmap(confs[best][modelo], annot=True, fmt='d', cmap='Greens')
        plt.title(f'{modelo} - Melhor Rodada')
        plt.subplot(1, 2, 2)
        sns.heatmap(confs[worst][modelo], annot=True, fmt='d', cmap='Reds')
        plt.title(f'{modelo} - Pior Rodada')
        plt.tight_layout()
        plt.show()

def plot_boxplot(resultados):
    plt.figure(figsize=(8, 6))
    plt.boxplot([extrair_acuracias(m, resultados) for m in resultados], labels=resultados.keys())
    plt.title("Boxplot - Acurácia dos Modelos")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.show()

def imprimir_estatisticas(resultados):
    for modelo in resultados:
        print(f'\nModelo: {modelo}')
        for k, v in estatisticas(modelo, resultados).items():
            print(f'{k}: {v:.4f}')


X, Y = carregar_dados()
resultados, confs = monte_carlo(X, Y)
plot_confusoes(confs, resultados)
imprimir_estatisticas(resultados)
plot_boxplot(resultados)