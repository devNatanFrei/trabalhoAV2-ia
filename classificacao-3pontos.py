import numpy as np
import matplotlib.pyplot as plt

# Custom train_test_split function
def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test

# 1. Organização do Conjunto de Dados
data = np.loadtxt('Spiral3d.csv', delimiter=',', skiprows=1)
X = data[:, :3]  # Features
y = data[:, 3]   # Labels
print(f'Dimensões de X: {X.shape}')
print(f'Dimensões de y: {y.shape}')

# 2. Visualização Inicial dos Dados
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], c='r', label='Classe 0')
ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], c='b', label='Classe 1')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.legend()
plt.savefig('initial_scatter.png')
plt.close()

# 3. Implementação dos Modelos de RNA

# Perceptron Simples
class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)
        for _ in range(self.n_epochs):
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

# Perceptron de Múltiplas Camadas (MLP)
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, n_epochs=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.loss_history = []  # Armazenar o erro por época

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y):
        for _ in range(self.n_epochs):
            hidden_input = np.dot(X, self.weights_input_hidden)
            hidden_output = self.sigmoid(hidden_input)
            output_input = np.dot(hidden_output, self.weights_hidden_output)
            output = self.sigmoid(output_input)
            error = y - output
            loss = np.mean((error) ** 2)
            self.loss_history.append(loss)  # Salva o erro médio da época
            d_output = error * self.sigmoid_derivative(output)
            error_hidden = d_output.dot(self.weights_hidden_output.T)
            d_hidden = error_hidden * self.sigmoid_derivative(hidden_output)
            self.weights_hidden_output += hidden_output.T.dot(d_output) * self.learning_rate
            self.weights_input_hidden += X.T.dot(d_hidden) * self.learning_rate

    def predict(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden)
        hidden_output = self.sigmoid(hidden_input)
        output_input = np.dot(hidden_output, self.weights_hidden_output)
        output = self.sigmoid(output_input)
        return np.round(output)

# Hiperparâmetros:
# Perceptron: learning_rate=0.01 (padrão), n_epochs=1000 (suficiente para convergência)
# MLP: hidden_size=10 (base), learning_rate=0.01 (padrão), n_epochs=1000 (base)

# 4. Identificação de Underfitting e Overfitting para MLP
def calculate_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    return accuracy, sensitivity, specificity

X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)


# Treinamento para visualização do erro
mlp_loss = MLP(input_size=3, hidden_size=10, output_size=1)
mlp_loss.fit(X_train, y_train.reshape(-1, 1))

# Gráfico do erro (loss) por época
plt.figure(figsize=(8, 5))
plt.plot(mlp_loss.loss_history, color='purple')
plt.title('Erro Quadrático Médio por Época (MLP)', fontsize=14)
plt.xlabel('Época', fontsize=12)
plt.ylabel('Erro Médio (Loss)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('mlp_loss_curve.png')
plt.show()

# Re-treinar modelos para capturar o histórico de loss
mlp_under_loss = MLP(input_size=3, hidden_size=2, output_size=1)
mlp_under_loss.fit(X_train, y_train.reshape(-1, 1))

mlp_over_loss = MLP(input_size=3, hidden_size=1000, output_size=1)
mlp_over_loss.fit(X_train, y_train.reshape(-1, 1))

# Plotar as curvas de erro para underfitting e overfitting
plt.figure(figsize=(10, 6))
plt.plot(mlp_under_loss.loss_history, label='Underfitting (2 neurônios)', color='red')
plt.plot(mlp_over_loss.loss_history, label='Overfitting (1000 neurônios)', color='blue')
plt.title('Curvas de Erro por Época - Underfitting vs Overfitting', fontsize=14)
plt.xlabel('Época', fontsize=12)
plt.ylabel('Erro Quadrático Médio (Loss)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('curvas_loss_under_over.png')
plt.show()

# Underfitting (2 neurônios)
mlp_under = MLP(input_size=3, hidden_size=2, output_size=1)
mlp_under.fit(X_train, y_train.reshape(-1, 1))
y_pred_train_under = mlp_under.predict(X_train).flatten()
y_pred_test_under = mlp_under.predict(X_test).flatten()
acc_train_under, sens_train_under, spec_train_under = calculate_metrics(y_train, y_pred_train_under)
acc_test_under, sens_test_under, spec_test_under = calculate_metrics(y_test, y_pred_test_under)
print(f'Underfitting - Acurácia Treino: {acc_train_under}, Teste: {acc_test_under}')

# Overfitting (1000 neurônios)
mlp_over = MLP(input_size=3, hidden_size=1000, output_size=1)
mlp_over.fit(X_train, y_train.reshape(-1, 1))
y_pred_train_over = mlp_over.predict(X_train).flatten()
y_pred_test_over = mlp_over.predict(X_test).flatten()
acc_train_over, sens_train_over, spec_train_over = calculate_metrics(y_train, y_pred_train_over)
acc_test_over, sens_test_over, spec_test_over = calculate_metrics(y_test, y_pred_test_over)
print(f'Overfitting - Acurácia Treino: {acc_train_over}, Teste: {acc_test_over}')

# Matriz de Confusão
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = np.zeros((2, 2))
    for i in range(len(y_true)):
        cm[int(y_true[i]), int(y_pred[i])] += 1
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ['0', '1'])
    plt.yticks([0, 1], ['0', '1'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    plt.close()

plot_confusion_matrix(y_test, y_pred_test_under, 'Confusion Matrix - Underfitting', 'cm_underfitting.png')
plot_confusion_matrix(y_test, y_pred_test_over, 'Confusion Matrix - Overfitting', 'cm_overfitting.png')

# 5. Validação com Simulações por Monte Carlo
R = 250
results_perceptron = {'accuracy': [], 'sensitivity': [], 'specificity': []}
results_mlp = {'accuracy': [], 'sensitivity': [], 'specificity': []}

for i in range(R):
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=i)
    
    # Perceptron
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)
    acc, sens, spec = calculate_metrics(y_test, y_pred)
    results_perceptron['accuracy'].append(acc)
    results_perceptron['sensitivity'].append(sens)
    results_perceptron['specificity'].append(spec)
    
    # MLP
    mlp = MLP(input_size=3, hidden_size=10, output_size=1)
    mlp.fit(X_train, y_train.reshape(-1, 1))
    y_pred = mlp.predict(X_test).flatten()
    acc, sens, spec = calculate_metrics(y_test, y_pred)
    results_mlp['accuracy'].append(acc)
    results_mlp['sensitivity'].append(sens)
    results_mlp['specificity'].append(spec)

# 6. Maior e Menor Acurácia para MLP
max_acc_idx = np.argmax(results_mlp['accuracy'])
min_acc_idx = np.argmin(results_mlp['accuracy'])

# Maior Acurácia
X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=max_acc_idx)
mlp_max = MLP(input_size=3, hidden_size=10, output_size=1)
mlp_max.fit(X_train, y_train.reshape(-1, 1))
y_pred_max = mlp_max.predict(X_test).flatten()
plot_confusion_matrix(y_test, y_pred_max, 'Confusion Matrix - Max Accuracy', 'cm_max_accuracy.png')

# Menor Acurácia
X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=min_acc_idx)
mlp_min = MLP(input_size=3, hidden_size=10, output_size=1)
mlp_min.fit(X_train, y_train.reshape(-1, 1))
y_pred_min = mlp_min.predict(X_test).flatten()
plot_confusion_matrix(y_test, y_pred_min, 'Confusion Matrix - Min Accuracy', 'cm_min_accuracy.png')

# 7. Cálculo das Estatísticas Finais
def print_statistics(results, model_name):
    print(f'Estatísticas para {model_name}:')
    for metric in ['accuracy', 'sensitivity', 'specificity']:
        values = results[metric]
        mean = np.mean(values)
        std = np.std(values)
        max_val = np.max(values)
        min_val = np.min(values)
        print(f'{metric.capitalize()}: Média: {mean:.4f}, Desvio-Padrão: {std:.4f}, Máx: {max_val:.4f}, Mín: {min_val:.4f}')

print_statistics(results_perceptron, 'Perceptron')
print_statistics(results_mlp, 'MLP')

# Gráfico 3D da espiral com classes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='coolwarm', edgecolor='k', alpha=0.8)
ax.set_title('Representação 3D da Espiral (Spiral3d)', fontsize=14)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
fig.colorbar(scatter, label='Classe')
plt.tight_layout()
plt.savefig('spiral3d_plot.png')
plt.show()

# Histogramas das métricas para Perceptron e MLP
metrics = ['accuracy', 'sensitivity', 'specificity']
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
for i, metric in enumerate(metrics):
    axes[i, 0].hist(results_perceptron[metric], bins=20, color='skyblue', edgecolor='black')
    axes[i, 0].set_title(f'{metric.capitalize()} - Perceptron')
    axes[i, 0].set_xlabel(metric)
    axes[i, 0].set_ylabel('Frequência')
    
    axes[i, 1].hist(results_mlp[metric], bins=20, color='salmon', edgecolor='black')
    axes[i, 1].set_title(f'{metric.capitalize()} - MLP')
    axes[i, 1].set_xlabel(metric)
    axes[i, 1].set_ylabel('Frequência')

plt.tight_layout()
plt.savefig('metricas_histogramas.png')
plt.show()
