import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 1. Leitura e organização dos dados
data = np.loadtxt('Spiral3d.csv', delimiter=',')
X = data[:, :3]
y = data[:, 3].astype(int)

# 2. Visualização inicial - gráfico de dispersão 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='bwr', alpha=0.6)
plt.title("Visualização inicial dos dados")
plt.show()

# Funções auxiliares
def train_test_split(X, y, test_size=0.2):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    split = int(X.shape[0] * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix_elements(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

def compute_metrics(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix_elements(y_true, y_pred)
    acc = (TP + TN) / (TP + TN + FP + FN)
    sens = TP / (TP + FN) if (TP + FN) else 0
    spec = TN / (TN + FP) if (TN + FP) else 0
    return acc, sens, spec

# 3. Perceptron Simples
class SimplePerceptron:
    def __init__(self, lr=0.01, n_iter=100):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.errors = []

        for _ in range(self.n_iter):
            error = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.weights += update * xi
                self.bias += update
                error += int(update != 0.0)
            self.errors.append(error)

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

# 4. MLP (1 hidden layer)
class MLP:
    def __init__(self, hidden_size=5, epochs=1000, lr=0.01):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y):
        self.input_size = X.shape[1]
        self.output_size = 1

        # Weight initialization
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

        self.loss_curve = []

        for epoch in range(self.epochs):
            # Forward
            z1 = X @ self.W1 + self.b1
            a1 = sigmoid(z1)
            z2 = a1 @ self.W2 + self.b2
            a2 = sigmoid(z2)

            # Loss (MSE)
            loss = np.mean((a2 - y.reshape(-1, 1))**2)
            self.loss_curve.append(loss)

            # Backpropagation
            d2 = (a2 - y.reshape(-1, 1)) * sigmoid_deriv(a2)
            d1 = d2 @ self.W2.T * sigmoid_deriv(a1)

            self.W2 -= self.lr * a1.T @ d2
            self.b2 -= self.lr * np.sum(d2, axis=0, keepdims=True)
            self.W1 -= self.lr * X.T @ d1
            self.b1 -= self.lr * np.sum(d1, axis=0, keepdims=True)

    def predict(self, X):
        a1 = sigmoid(X @ self.W1 + self.b1)
        a2 = sigmoid(a1 @ self.W2 + self.b2)
        return (a2 > 0.5).astype(int).ravel()

# 5. RBF Network
class RBF:
    def __init__(self, num_centers=10, lr=0.01):
        self.num_centers = num_centers
        self.lr = lr

    def _kernel(self, X, C, beta=1.0):
        diff = X[:, np.newaxis] - C
        return np.exp(-beta * np.sum(diff ** 2, axis=2))

    def fit(self, X, y):
        rand_idx = np.random.choice(X.shape[0], self.num_centers, replace=False)
        self.centers = X[rand_idx]
        G = self._kernel(X, self.centers)
        self.weights = np.linalg.pinv(G) @ y

    def predict(self, X):
        G = self._kernel(X, self.centers)
        output = G @ self.weights
        return (output > 0.5).astype(int)

# 6. Validação Monte Carlo
R = 250
results = {"perceptron": [], "mlp": [], "rbf": []}
best, worst = {"acc": 0}, {"acc": 1}

for i in range(R):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Perceptron
    p = SimplePerceptron()
    p.fit(X_train, y_train)
    pred_p = p.predict(X_test)
    acc_p, sens_p, spec_p = compute_metrics(y_test, pred_p)

    # MLP
    m = MLP(hidden_size=10)
    m.fit(X_train, y_train)
    pred_m = m.predict(X_test)
    acc_m, sens_m, spec_m = compute_metrics(y_test, pred_m)

    # RBF
    r = RBF(num_centers=20)
    r.fit(X_train, y_train)
    pred_r = r.predict(X_test)
    acc_r, sens_r, spec_r = compute_metrics(y_test, pred_r)

    # Store
    results["perceptron"].append((acc_p, sens_p, spec_p))
    results["mlp"].append((acc_m, sens_m, spec_m))
    results["rbf"].append((acc_r, sens_r, spec_r))

    if acc_m > best["acc"]:
        best.update({"acc": acc_m, "model": m, "pred": pred_m, "true": y_test, "loss": m.loss_curve})
    if acc_m < worst["acc"]:
        worst.update({"acc": acc_m, "model": m, "pred": pred_m, "true": y_test, "loss": m.loss_curve})

# 7. Estatísticas finais
def summarize(metric_index):
    def stats(model_name):
        vals = [r[metric_index] for r in results[model_name]]
        return {
            "mean": np.mean(vals),
            "std": np.std(vals),
            "max": np.max(vals),
            "min": np.min(vals)
        }
    return stats("perceptron"), stats("mlp"), stats("rbf")

acc_stats = summarize(0)
sens_stats = summarize(1)
spec_stats = summarize(2)

print("ACURÁCIA:", acc_stats)
print("SENSIBILIDADE:", sens_stats)
print("ESPECIFICIDADE:", spec_stats)

# Plot curvas de aprendizado
plt.plot(best["loss"], label='Melhor Acurácia')
plt.plot(worst["loss"], label='Pior Acurácia')
plt.title("Curva de aprendizado (MLP)")
plt.xlabel("Épocas")
plt.ylabel("Erro")
plt.legend()
plt.show()
