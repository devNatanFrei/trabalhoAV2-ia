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
    plt.show()


class Adaline:
    def __init__(self, X_treino, Y_treino, num_max_epoca, taxa_aprendizagem, precisao, plot=True):
        self.p, self.N = X_treino.shape  
        self.X_treino = np.vstack((-np.ones((1, self.N)), X_treino))  
        self.Y_treino = Y_treino
        self.taxa_aprendizagem = taxa_aprendizagem
        self.pr = precisao
        self.num_max_e = num_max_epoca
        self.w = np.random.random_sample((self.p + 1, 1))  
        self.plot = plot
        self.x1 = np.linspace(-2, 7)
        
        if self.plot:
            plt.figure(1)
            plt.scatter(X_treino[0, :5], X_treino[1, :5], c='purple')
            plt.scatter(X_treino[0, 5:], X_treino[1, 5:], c='blue')
            self.linha = plt.plot(self.x1, self.__gerar_reta(), c='k')
            plt.xlim(-0.5, 6.5)
            plt.ylim(-0.5, 6.5)

    def treino(self):
        epocas = 0
        EQM1 = 1
        EQM2 = 0
        hist_eqm = []

        while epocas < self.num_max_e and abs(EQM1 - EQM2) > self.pr:
            EQM1 = self.__EQM(self.X_treino)
            hist_eqm.append(EQM1)

            for k in range(self.N):
                x_k = self.X_treino[:, k].reshape(self.p + 1, 1)
                u_k = float(self.w.T @ x_k)
                d_k = float(self.Y_treino[k])
                e_k = d_k - u_k
                self.w += self.taxa_aprendizagem * e_k * x_k

            if self.plot:
                plt.pause(0.1)
                self.linha[0].remove()
                self.linha = plt.plot(self.x1, self.__gerar_reta(), c='k')

            EQM2 = self.__EQM(self.X_treino)
            epocas += 1

        print(f"Épocas até convergência: {epocas}")
        
        if self.plot:
            plt.figure(2)
            plt.plot(hist_eqm)
            plt.xlabel("Épocas")
            plt.ylabel("EQM")
            plt.title("Curva de Convergência do EQM")
            plt.grid(True)
            plt.show()

    def predizer(self, X):
        X_bias = np.vstack((-np.ones((1, X.shape[1])), X))
        y_pred = self.w.T @ X_bias
        return y_pred

    def __EQM(self, X):
        eqm = 0
        for k in range(self.N):
            x_k = X[:, k].reshape(self.p + 1, 1)
            u_k = float(self.w.T @ x_k)
            d_k = float(self.Y_treino[k])
            eqm += (d_k - u_k) ** 2
        return eqm / (2 * self.N)

    def __gerar_reta(self):
        
        return np.nan_to_num(-self.x1 * self.w[1, 0] / self.w[2, 0] + self.w[0, 0] / self.w[2, 0])
    
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



print("Monte Carlo Validation Results")
print()

results_adaline = monte_carlo_validation(Adaline, x, y, R=250, learning_rate=0.01, n_ephocs=1000)
results_mlp_under = monte_carlo_validation(MLP, x, y, R=250, n_hidden=3, learning_rate=0.01, n_ephocs=5000, random_state=42)
results_mlp_over = monte_carlo_validation(MLP, x, y, R=250, n_hidden=50, learning_rate=0.01, n_ephocs=5000, random_state=42)

print("Adaline Results:", results_adaline)
print("MLP Underfitting Results:", results_mlp_under)
print("MLP Overfitting Results:", results_mlp_over)