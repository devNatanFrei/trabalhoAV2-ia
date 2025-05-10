import numpy as np
import matplotlib.pyplot as plt

# Carregamento dos dados
def load_data():
    data = np.loadtxt('aerogerador.dat', delimiter='\t')
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    return x, y

# Gráfico de dispersão dos dados
def plot_data(x, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=10)
    plt.xlabel('Velocidade do Vento')
    plt.ylabel('Potência Gerada')
    plt.title('Potência Gerada pelo Aerogerador')

# Modelo ADALINE
class Adaline:
    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.w = np.random.randn(n_features, 1)
        self.b = 0
        self.losses = []

        for _ in range(self.n_epochs):
            y_pred = x @ self.w + self.b
            error = y - y_pred
            mse = np.mean(error ** 2)
            self.losses.append(mse)

            self.w += self.learning_rate * (x.T @ error) / n_samples
            self.b += self.learning_rate * np.mean(error)

    def predict(self, x):
        return x @ self.w + self.b

# Modelo MLP com 1 camada oculta
class MLP:
    def __init__(self, n_hidden, learning_rate, n_ephocs, random_state=None):
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.n_ephocs = n_ephocs
        self.random_state = random_state

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def fit(self, x, y):
        if self.random_state:
            np.random.seed(self.random_state)
        
        n_samples, n_features = x.shape
        self.w1 = np.random.rand(n_features, self.n_hidden)
        self.b1 = np.zeros((1, self.n_hidden))
        self.w2 = np.random.rand(self.n_hidden, 1)
        self.b2 = np.zeros((1, 1))
        self.losses = []
        
        for _ in range(self.n_ephocs):
            z1 = x @ self.w1 + self.b1
            a1 = self.sigmoid(z1)
            z2 = a1 @ self.w2 + self.b2
            output = z2
            
            error = y - output
            mse = np.mean(error ** 2)
            self.losses.append(mse)
            
            d_output = -2 * error
            d_w2 = (a1.T @ d_output) / n_samples
            d_b2 = np.mean(d_output, axis=0, keepdims=True)
            
            d_a1 = d_output @ self.w2.T
            d_z1 = d_a1 * self.sigmoid_derivative(a1)
            d_w1 = (x.T @ d_z1) / n_samples
            d_b1 = np.mean(d_z1, axis=0, keepdims=True)
            
            self.w1 -= self.lr * d_w1
            self.b1 -= self.lr * d_b1   
            self.w2 -= self.lr * d_w2
            self.b2 -= self.lr * d_b2
            
    def predict(self, x):
        a1 = self.sigmoid(x @ self.w1 + self.b1)
        output = a1 @ self.w2 + self.b2
        return output

# Curvas de aprendizado
def plot_learning_curve(models, labels, title='Curvas de Aprendizado'):
    plt.figure(figsize=(10, 6))
    for model, label in zip(models, labels):
        plt.plot(model.losses, label=label)
    plt.title(title)
    plt.xlabel('Épocas')
    plt.ylabel('Erro Quadrático Médio (MSE)')
    plt.legend()
    plt.grid()

# Validação Monte Carlo (250 rodadas)
def monte_carlo_validation(model_class, x, y, R, **model_params):
    mse_list = []
    
    for _ in range(R):
        indic = np.random.permutation(x.shape[0])
        n_train = int(0.8 * x.shape[0])
        train_indic, test_indic = indic[:n_train], indic[n_train:]
        
        x_train, y_train = x[train_indic], y[train_indic]
        x_test, y_test = x[test_indic], y[test_indic]
        
        model = model_class(**model_params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        mse = np.mean((y_test - y_pred) ** 2)
        mse_list.append(mse)
        
    mse_list = np.array(mse_list)
    
    results = {
        'mean': mse_list.mean(),
        'std':  mse_list.std(),
        'min':  mse_list.min(),
        'max':  mse_list.max(),
    }
    return results

# Execução principal
x, y = load_data()
plot_data(x, y)
plt.show()

# Comentário: Hiperparâmetros foram ajustados empiricamente com base na estabilidade da curva de aprendizado.

# Treinamento MLP com 3 topologias
mlp_under = MLP(n_hidden=2, learning_rate=0.01, n_ephocs=600)      # underfitting
mlp_ideal = MLP(n_hidden=10, learning_rate=0.01, n_ephocs=600)     # adequada
mlp_over  = MLP(n_hidden=50, learning_rate=0.01, n_ephocs=600)     # overfitting

mlp_under.fit(x, y)
mlp_ideal.fit(x, y)
mlp_over.fit(x, y)

# Gráfico de curvas de aprendizado
plot_learning_curve([mlp_under, mlp_ideal, mlp_over],
                    ['Underfitting (2 neurônios)', 'Adequada (10 neurônios)', 'Overfitting (50 neurônios)'],
                    title='Curvas de Aprendizado - MLP')
plt.show()

# Validação Monte Carlo
adaline_results = monte_carlo_validation(Adaline, x, y, R=250, learning_rate=0.01, n_epochs=1000)
mlp_results     = monte_carlo_validation(MLP, x, y, R=250, n_hidden=10, learning_rate=0.01, n_ephocs=1000)

# Impressão dos resultados
print("\nResultados da Validação Monte Carlo (250 rodadas):")
print("ADALINE:")
for k, v in adaline_results.items():
    print(f"  {k}: {v:.6f}")

print("\nMLP (n_hidden=10):")
for k, v in mlp_results.items():
    print(f"  {k}: {v:.6f}")
