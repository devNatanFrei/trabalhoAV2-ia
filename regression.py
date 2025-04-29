import numpy as np
import matplotlib.pyplot as plt

def load_data():
    data = np.loadtxt('aerogerador.dat', delimiter='\t')
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    return x, y

def plot_data(x, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=10)
    plt.xlabel('Velocidade do Vento')
    plt.ylabel('Potência Gerada')
    plt.title('Potência Gerada')
    # plt.show()


class Adaline:
    def __init__(self, learning_rate, n_ephocs):
        self.lr = learning_rate
        self.n_ephocs = n_ephocs
        
    
    def fit(self, x, y):
        self.w = np.zeros((x.shape[1] + 1, 1))
        x_bias = np.hstack((np.ones((x.shape[0], 1)), x))
        self.losses = []

        for _ in range(self.n_ephocs):
            y_pred = x_bias @ self.w
            error = y - y_pred
            self.w += self.lr * (x_bias.T @ error) / x.shape[0]
            mse = np.mean(error ** 2)
            self.losses.append(mse)

    def predict(self, x):
        x_bias = np.hstack((np.ones((x.shape[0], 1)), x))
        return x_bias @ self.w  
    
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
            
            d_output = -2*error
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
    

def plot_learning_curve(models, labels, title='Learning Curve'):
    for model, label in zip(models, labels):
        plt.plot(model.losses, label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid()

   

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
     
     
x, y = load_data()
plot_data(x, y)

adaline = Adaline(learning_rate=0.01, n_ephocs=1000)
adaline.fit(x, y)

mlp_under = MLP(n_hidden=3, learning_rate=0.01, n_ephocs=5000, random_state=42)
mlp_under.fit(x, y)

mlp_over = MLP(n_hidden=50, learning_rate=0.01, n_ephocs=5000, random_state=42)
mlp_over.fit(x, y)

# plot_learning_curve(
#     [adaline, mlp_under, mlp_over],  
#     ['Adaline', 'MLP Underfitting', 'MLP Overfitting'], 
#     title='Learning Curve'
# )

print("Monte Carlo Validation Results")
print()

results_adaline = monte_carlo_validation(Adaline, x, y, R=250, learning_rate=0.01, n_ephocs=1000)
results_mlp_under = monte_carlo_validation(MLP, x, y, R=250, n_hidden=3, learning_rate=0.01, n_ephocs=5000, random_state=42)
results_mlp_over = monte_carlo_validation(MLP, x, y, R=250, n_hidden=50, learning_rate=0.01, n_ephocs=5000, random_state=42)

print("Adaline Results:", results_adaline)
print("MLP Underfitting Results:", results_mlp_under)
print("MLP Overfitting Results:", results_mlp_over)