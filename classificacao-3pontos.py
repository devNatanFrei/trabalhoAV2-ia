import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Configurações para reprodutibilidade
np.random.seed(42)

# --- Funções Auxiliares ---

def map_labels_to_01(y_arr):
    """Mapeia rótulos de duas classes para 0 e 1."""
    unique_labels = np.unique(y_arr)
    if len(unique_labels) > 2: # Permite dataset com apenas uma classe (para debug)
        # Se for para produção, melhor levantar erro se não forem 2 classes
        print(f"Aviso: Esperava no máximo 2 classes, encontrou {len(unique_labels)}: {unique_labels}")
        if len(unique_labels) == 1: # Se só uma classe, mapeia para 0 ou 1
             print(f"Mapeando a única classe {unique_labels[0]} para 0.")
             return np.zeros_like(y_arr, dtype=int)

    if len(unique_labels) == 1: # Se já era só uma classe e foi tratada acima
        return y_arr.astype(int)

    # Se já são 0 e 1, não faz nada (garante que sejam inteiros)
    if np.array_equal(np.sort(unique_labels), np.array([0,1])):
        return y_arr.astype(int)
    
    if len(unique_labels) == 2:
        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
        print(f"Mapeando rótulos originais {unique_labels} para: {label_map}")
        return np.array([label_map[label] for label in y_arr]).astype(int)
    else: # Caso de 1 classe não coberto acima, ou 0 classes
        print(f"Aviso: Não foi possível mapear rótulos {unique_labels} para 0 e 1.")
        return y_arr.astype(int) # Retorna como está, pode causar problemas depois


def load_data(filepath):
    """Carrega os dados do arquivo CSV."""
    try:
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    except Exception as e:
        print(f"Erro ao carregar {filepath}: {e}")
        print("Verifique se o arquivo CSV existe, tem o delimitador ',' e o cabeçalho é apenas a primeira linha.")
        return None, None

    if data.shape[1] < 4:
        print(f"Erro: O arquivo CSV deve ter pelo menos 4 colunas. Encontrado: {data.shape[1]}")
        return None, None
        
    X = data[:, :3]
    y_raw = data[:, 3] # Carrega como está para inspeção e mapeamento
    return X, y_raw

def standardize_data(X):
    """Padroniza os dados (média 0, desvio padrão 1)."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Adicionar epsilon para evitar divisão por zero se std for 0 para alguma feature
    return (X - mean) / (std + 1e-8), mean, std

def add_bias(X):
    """Adiciona o termo de bias (coluna de 1s) a X."""
    return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

def train_test_split_manual(X, y, test_size=0.2, random_state=None):
    """Divide manualmente os dados em treino e teste."""
    if random_state is not None:
        # Define a semente apenas para esta operação, se um estado aleatório for fornecido
        # Isso é útil para divisões consistentes fora do loop de Monte Carlo
        current_random_state = np.random.get_state()
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - test_size))
    
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    if random_state is not None:
        np.random.set_state(current_random_state) # Restaura o estado aleatório global
        
    return X_train, X_test, y_train, y_test


def sigmoid(z):
    """Função de ativação Sigmoid."""
    return 1 / (1 + np.exp(-np.clip(z, -700, 700))) # Clip para evitar overflow/underflow extremo

def sigmoid_derivative(z_net_input): # Renomeado para clareza que recebe o net input
    """Derivada da função Sigmoid, em termos da SAÍDA da sigmoid s = sigmoid(z_net_input)."""
    s = sigmoid(z_net_input)
    return s * (1 - s)
    # Alternativamente, se você já tem 's' (a ativação), a derivada é s * (1-s)
    # A implementação atual recalcula s. Se 's' já estiver disponível, use-o.

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calcula TP, TN, FP, FN, acurácia, sensibilidade, especificidade."""
    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(y_pred_proba, np.ndarray): y_pred_proba = np.array(y_pred_proba)

    # Garantir que y_true seja 0 ou 1
    if not (np.all(np.isin(y_true, [0, 1]))):
         print(f"Aviso em calculate_metrics: y_true contém valores diferentes de 0 ou 1: {np.unique(y_true)}")
         # Poderia tentar converter ou levantar erro. Por agora, aviso.

    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Debug:
    print(f"calculate_metrics -> y_true unique: {np.unique(y_true, return_counts=True)}")
    print(f"calculate_metrics -> y_pred_proba (first 5): {y_pred_proba[:5]}")
    print(f"calculate_metrics -> y_pred unique: {np.unique(y_pred, return_counts=True)}")

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return tp, tn, fp, fn, accuracy, sensitivity, specificity

def confusion_matrix_manual(y_true, y_pred_proba, threshold=0.5):
    """Retorna a matriz de confusão [TN, FP], [FN, TP]."""
    tp, tn, fp, fn, _, _, _ = calculate_metrics(y_true, y_pred_proba, threshold)
    return np.array([[tn, fp], [fn, tp]])

def plot_confusion_matrix(cm, title='Matriz de Confusão', class_names=['Classe 0', 'Classe 1']):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='Rótulo Verdadeiro',
           xlabel='Rótulo Predito')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(f'imgs/{title}.png', dpi=300)
    plt.show()


def plot_learning_curve(train_losses, val_losses=None, title='Curva de Aprendizado', loss_label='Perda'):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label=f'Perda no Treinamento ({loss_label})')
    if val_losses:
        plt.plot(val_losses, label=f'Perda na Validação ({loss_label})')
    plt.xlabel('Épocas')
    plt.ylabel(loss_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'imgs/{title}.png', dpi=300)
    plt.show()

# --- 1. Organização do Conjunto de Dados ---
print("1. Organização do Conjunto de Dados")
filepath = 'Spiral3d.csv' # Certifique-se que este arquivo está no mesmo diretório
X_orig_raw, y_orig_raw = load_data(filepath)

if X_orig_raw is None:
    print("Falha ao carregar dados. Encerrando.")
    exit()

# Mapear rótulos para 0 e 1
y_orig = map_labels_to_01(y_orig_raw)
print(f"Valores únicos e contagens em y_orig (APÓS MAPEAMENTO): {np.unique(y_orig, return_counts=True)}")

# Padronização dos dados
X_scaled, mean_X, std_X = standardize_data(X_orig_raw) # Usar X_orig_raw aqui

# Adicionar termo de bias a X_scaled (será usado pelos modelos)
# X_biased é usado por Perceptron e MLP no FIT e PREDICT
# Não vamos mais usar X_biased globalmente, será criado dentro do fit/predict ou antes do split.

# Target y precisa ser um vetor coluna para algumas operações matriciais
y_column_vector = y_orig.reshape(-1, 1)

print(f"Shape original de X: {X_orig_raw.shape}")
print(f"Shape original de y (mapeado): {y_orig.shape}")
print(f"Shape de X padronizado: {X_scaled.shape}")
print(f"Shape de y como vetor coluna: {y_column_vector.shape}")
print("Classes (após mapeamento):", np.unique(y_orig))
print("-" * 30)

# --- 2. Visualização Inicial dos Dados ---
print("2. Visualização Inicial dos Dados")
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# Usar y_orig (mapeado) para colorir
scatter = ax.scatter(X_orig_raw[:, 0], X_orig_raw[:, 1], X_orig_raw[:, 2], c=y_orig, cmap='viridis', marker='o')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('Visualização 3D do Conjunto de Dados Spiral3d (Rótulos Mapeados)')
if len(np.unique(y_orig)) == 2 :
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
else:
    print("Aviso: Não foi possível gerar legenda para o scatter plot (número de classes != 2).")
plt.savefig('imgs/spiral3d_3dscatter.png', dpi=300)
plt.show()
print("-" * 30)

# --- 3. Modelos de RNA ---

# --- Perceptron Simples ---
print("3. Modelos de RNA: Perceptron Simples")

class SimplePerceptron:
    def __init__(self, learning_rate=0.01, n_epochs=600):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.errors_ = [] 

    def fit(self, X_biased_train, y_train): # y_train deve ser 0 ou 1
        # Converter y_train (0,1) para y_perceptron (-1,1) para o algoritmo
        y_perceptron_train = np.where(y_train == 0, -1, 1)
        
        # Inicialização de pesos (pequenos, centrados em zero)
        self.weights = (np.random.rand(X_biased_train.shape[1]) - 0.5) * 0.02 

        for _ in range(self.n_epochs):
            errors = 0
            for xi, target_perceptron in zip(X_biased_train, y_perceptron_train):
                # net_input_val = self.net_input(xi) # Não precisa aqui
                prediction_perceptron = self._predict_raw_perceptron_output(xi) # Retorna -1 ou 1
                update = self.lr * (target_perceptron - prediction_perceptron) # update pode ser -2*lr, 0, 2*lr
                if update != 0: # Otimização: só atualiza se houver erro
                    self.weights += update * xi
                    errors += 1
            self.errors_.append(errors)
            if errors == 0 and self.n_epochs > 1: # Convergiu (para dados linearmente separáveis)
                # print(f"Perceptron convergiu na época {_+1}") # Opcional
                break 
        return self

    def _net_input(self, X_biased_row):
        return np.dot(X_biased_row, self.weights)

    def _predict_raw_perceptron_output(self, X_biased_row): # Saída do Perceptron (-1 ou 1)
        return np.where(self._net_input(X_biased_row) >= 0.0, 1, -1)

    def predict(self, X_biased_test): # Saída final (0 ou 1)
        raw_predictions_perceptron = np.array([self._predict_raw_perceptron_output(xi) for xi in X_biased_test])
        return np.where(raw_predictions_perceptron == -1, 0, 1) # Converte -1 para 0, 1 para 1

    def predict_proba(self, X_biased_test): # Simula proba para compatibilidade
        # A "probabilidade" do Perceptron é um artifício.
        # Usamos a saída da net_input passada pela sigmoid.
        net_inputs = np.array([self._net_input(xi) for xi in X_biased_test])
        return sigmoid(net_inputs)


# Divisão para treino do Perceptron (fora do Monte Carlo, apenas para demonstração)
X_train_sp_demo, X_test_sp_demo, y_train_sp_demo, y_test_sp_demo = train_test_split_manual(X_scaled, y_orig, test_size=0.2, random_state=123)
X_train_sp_demo_biased = add_bias(X_train_sp_demo)
X_test_sp_demo_biased = add_bias(X_test_sp_demo)

print(f"Demo Perceptron - y_train unique: {np.unique(y_train_sp_demo, return_counts=True)}")
print(f"Demo Perceptron - y_test unique: {np.unique(y_test_sp_demo, return_counts=True)}")


print("Discussão Hiperparâmetros Perceptron Simples:")
print(" - Taxa de Aprendizado (learning_rate): Escolhida como 0.01. Ajustes podem ser necessários.")
print(" - Número de Épocas (n_epochs): Escolhido como 600. Para Spiral3D, o Perceptron não convergirá para erro zero.")

perceptron_model_demo = SimplePerceptron(learning_rate=0.01, n_epochs=600)
perceptron_model_demo.fit(X_train_sp_demo_biased, y_train_sp_demo)

y_pred_proba_sp_demo = perceptron_model_demo.predict_proba(X_test_sp_demo_biased)
tp_sp, tn_sp, fp_sp, fn_sp, acc_sp, sens_sp, spec_sp = calculate_metrics(y_test_sp_demo, y_pred_proba_sp_demo)
cm_sp = confusion_matrix_manual(y_test_sp_demo, y_pred_proba_sp_demo)

print("\nDesempenho Perceptron Simples (em um split de exemplo):")
print(f"  Acurácia: {acc_sp:.8f}")
print(f"  Sensibilidade: {sens_sp:.8f}")
print(f"  Especificidade: {spec_sp:.8f}")
print("  Matriz de Confusão:")
print(cm_sp)
if sum(sum(cm_sp)) > 0 : plot_confusion_matrix(cm_sp, title="Matriz de Confusão - Perceptron Simples (Exemplo)")

plt.figure(figsize=(8,6))
plt.plot(range(1, len(perceptron_model_demo.errors_) + 1), perceptron_model_demo.errors_, marker='o')
plt.xlabel('Épocas')
plt.ylabel('Número de Atualizações de Peso (Erros no Treino)')
plt.title('Curva de Aprendizado - Perceptron Simples (Erros de Treino)')
plt.grid(True)
plt.savefig('imgs/spiral3d_perceptron_learning_curve.png', dpi=300)
plt.show()
print("-" * 30)


# --- MLP (Perceptron de Múltiplas Camadas) ---
print("MLP (Perceptron de Múltiplas Camadas)")

class MLP:
    def __init__(self, n_input_biased, n_hidden, n_output, learning_rate=0.01, epochs=100):
        self.n_input_biased = n_input_biased # Número de inputs incluindo o bias
        self.n_hidden = n_hidden
        self.n_output = n_output # Geralmente 1 para classificação binária com sigmoid
        self.lr = learning_rate
        self.epochs = epochs

        # Inicialização de pesos (Xavier/Glorot para sigmoid)
        # Camada Entrada -> Oculta
        fan_in_ih = self.n_input_biased
        fan_out_ih = self.n_hidden
        limit_ih = np.sqrt(6 / (fan_in_ih + fan_out_ih))
        self.weights_ih = np.random.uniform(-limit_ih, limit_ih, (fan_in_ih, self.n_hidden))
        
        # Camada Oculta -> Saída
        fan_in_ho = self.n_hidden + 1 # +1 para o bias da camada oculta
        fan_out_ho = self.n_output
        limit_ho = np.sqrt(6 / (fan_in_ho + fan_out_ho)) # CORRIGIDO
        self.weights_ho = np.random.uniform(-limit_ho, limit_ho, (fan_in_ho, self.n_output))

        self.train_losses = []
        self.val_losses = []

    def _forward(self, X_biased_input):
        # Camada de entrada para oculta
        # X_biased_input já deve ter o bias
        hidden_net_input = np.dot(X_biased_input, self.weights_ih)
        hidden_activation = sigmoid(hidden_net_input)
        
        # Adicionar bias à saída da camada oculta
        hidden_activation_biased = add_bias(hidden_activation)
        
        # Camada oculta para saída
        final_net_input = np.dot(hidden_activation_biased, self.weights_ho)
        final_activation_output = sigmoid(final_net_input) # Probabilidade para classe 1
        
        return hidden_activation_biased, final_activation_output, hidden_net_input, final_net_input

    def _binary_cross_entropy(self, y_true_col, y_pred_proba_col):
        epsilon = 1e-12 # Evitar log(0)
        y_pred_proba_col = np.clip(y_pred_proba_col, epsilon, 1 - epsilon)
        loss = -np.mean(y_true_col * np.log(y_pred_proba_col) + (1 - y_true_col) * np.log(1 - y_pred_proba_col))
        return loss

    def fit(self, X_train_biased, y_train_col, X_val_biased=None, y_val_col=None):
        self.train_losses = []
        self.val_losses = []

        if not (np.all(np.isin(y_train_col, [0, 1]))):
            print(f"Aviso em MLP.fit: y_train_col contém valores diferentes de 0 ou 1: {np.unique(y_train_col)}")
        if X_val_biased is not None and not (np.all(np.isin(y_val_col, [0, 1]))):
             print(f"Aviso em MLP.fit: y_val_col contém valores diferentes de 0 ou 1: {np.unique(y_val_col)}")


        for epoch in range(self.epochs):
            # Forward propagation
            # hidden_act_biased: ativações da camada oculta COM bias adicionado
            # final_act_train: ativações da camada de saída (probabilidades)
            # hidden_net_in: net input da camada oculta
            # final_net_in: net input da camada de saída
            hidden_act_biased, final_act_train, hidden_net_in, final_net_in = self._forward(X_train_biased)
            
            train_loss = self._binary_cross_entropy(y_train_col, final_act_train)
            self.train_losses.append(train_loss)

            # Backward propagation
            # Erro na camada de saída (delta para a camada de SAÍDA, não o erro da loss)
            # Para sigmoid + BCE, o delta (dL/d_net_input_saida) é (prediction - target)
            delta_output_layer = final_act_train - y_train_col 
            
            # Erro na camada oculta
            # Propagar delta_output_layer para a camada oculta, multiplicando pelos pesos (sem o bias da oculta)
            # e pela derivada da ativação da camada oculta.
            # error_at_hidden_output = delta_output_layer @ self.weights_ho[:-1, :].T
            error_propagated_to_hidden = np.dot(delta_output_layer, self.weights_ho[:-1, :].T) # Exclui o peso do bias da camada oculta
            delta_hidden_layer = error_propagated_to_hidden * sigmoid_derivative(hidden_net_in) # Usa net_input da oculta

            # Atualização de pesos
            # Oculta -> Saída: gradiente é hidden_act_biased.T @ delta_output_layer
            grad_weights_ho = np.dot(hidden_act_biased.T, delta_output_layer)
            self.weights_ho -= self.lr * grad_weights_ho / X_train_biased.shape[0]
            
            # Entrada -> Oculta: gradiente é X_train_biased.T @ delta_hidden_layer
            grad_weights_ih = np.dot(X_train_biased.T, delta_hidden_layer)
            self.weights_ih -= self.lr * grad_weights_ih / X_train_biased.shape[0]
            
            if X_val_biased is not None and y_val_col is not None:
                _, final_act_val, _, _ = self._forward(X_val_biased)
                val_loss = self._binary_cross_entropy(y_val_col, final_act_val)
                self.val_losses.append(val_loss)

            if (epoch + 1) % (max(1, self.epochs // 10)) == 0 or epoch == 0:
                 log_msg = f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.8f}"
                 if X_val_biased is not None and self.val_losses: # Checa se val_losses não está vazio
                    log_msg += f", Val Loss: {self.val_losses[-1]:.8f}" # Pega a última val_loss calculada
                 print(log_msg)
        return self

    def predict_proba(self, X_biased_test):
        _, final_activation_output, _, _ = self._forward(X_biased_test)
        return final_activation_output

    def predict(self, X_biased_test, threshold=0.5):
        proba = self.predict_proba(X_biased_test)
        return (proba >= threshold).astype(int)

# Preparar dados para MLP (dividindo uma vez para estudo de under/overfitting)
# Usar X_scaled e y_orig
X_train_mlp_demo, X_val_mlp_demo, y_train_mlp_demo, y_val_mlp_demo = train_test_split_manual(X_scaled, y_orig, test_size=0.2, random_state=42)
X_train_mlp_demo_biased = add_bias(X_train_mlp_demo)
X_val_mlp_demo_biased = add_bias(X_val_mlp_demo)
y_train_mlp_demo_col = y_train_mlp_demo.reshape(-1, 1)
y_val_mlp_demo_col = y_val_mlp_demo.reshape(-1, 1)

print(f"Demo MLP - y_train unique: {np.unique(y_train_mlp_demo, return_counts=True)}")
print(f"Demo MLP - y_val unique: {np.unique(y_val_mlp_demo, return_counts=True)}")


# --- 4. Underfitting e Overfitting para MLP ---
print("\n4. Underfitting e Overfitting para MLP")
print("Discussão Hiperparâmetros MLP:")
print(" - Neurônios Camada Oculta: Poucos -> underfitting. Muitos -> overfitting.")
print(" - Taxa de Aprendizado: Importante. Comum 0.001-0.1.")
print(" - Épocas: Muitas épocas com modelo complexo -> overfitting.")

n_features_biased_mlp = X_train_mlp_demo_biased.shape[1]
n_output_mlp = 1 

# Caso de Underfitting (MLP Subdimensionado)
print("\n--- MLP: Caso de Underfitting ---")
mlp_underfit = MLP(n_input_biased=n_features_biased_mlp, n_hidden=2, n_output=n_output_mlp, learning_rate=0.1, epochs=300) # Aumentar épocas um pouco
print("Topologia MLP Underfitting: 1 camada oculta com 2 neurônios.")
mlp_underfit.fit(X_train_mlp_demo_biased, y_train_mlp_demo_col, X_val_mlp_demo_biased, y_val_mlp_demo_col)

y_pred_proba_under = mlp_underfit.predict_proba(X_val_mlp_demo_biased)
tp_u, tn_u, fp_u, fn_u, acc_u, sens_u, spec_u = calculate_metrics(y_val_mlp_demo, y_pred_proba_under)
cm_u = confusion_matrix_manual(y_val_mlp_demo, y_pred_proba_under)

print("\nDesempenho MLP Underfitting (Validação):")
print(f"  Acurácia: {acc_u:.8f}")
print(f"  Sensibilidade: {sens_u:.8f}")
print(f"  Especificidade: {spec_u:.8f}")
print("  Matriz de Confusão:")
print(cm_u)
if sum(sum(cm_u)) > 0 : plot_confusion_matrix(cm_u, title="Matriz de Confusão - MLP Underfitting")
plot_learning_curve(mlp_underfit.train_losses, mlp_underfit.val_losses, title="Curva de Aprendizado - MLP Underfitting", loss_label="BCE")

# Caso de "Bom Ajuste" (MLP Bem Dimensionado)
print("\n--- MLP: Caso de Bom Ajuste (Referência) ---")
neurons_good_fit = 20 # Aumentado para Spiral3D
epochs_good_fit = 700 # Aumentado
lr_good_fit = 0.05    # Reduzido um pouco
mlp_goodfit = MLP(n_input_biased=n_features_biased_mlp, n_hidden=neurons_good_fit, n_output=n_output_mlp, 
                  learning_rate=lr_good_fit, epochs=epochs_good_fit)
print(f"Topologia MLP Bom Ajuste: 1 camada oculta com {neurons_good_fit} neurônios.")
mlp_goodfit.fit(X_train_mlp_demo_biased, y_train_mlp_demo_col, X_val_mlp_demo_biased, y_val_mlp_demo_col)

y_pred_proba_good = mlp_goodfit.predict_proba(X_val_mlp_demo_biased)
tp_g, tn_g, fp_g, fn_g, acc_g, sens_g, spec_g = calculate_metrics(y_val_mlp_demo, y_pred_proba_good)
cm_g = confusion_matrix_manual(y_val_mlp_demo, y_pred_proba_good)

print("\nDesempenho MLP Bom Ajuste (Validação):")
print(f"  Acurácia: {acc_g:.8f}")
print(f"  Sensibilidade: {sens_g:.8f}")
print(f"  Especificidade: {spec_g:.8f}")
print("  Matriz de Confusão:")
print(cm_g)
if sum(sum(cm_g)) > 0 : plot_confusion_matrix(cm_g, title="Matriz de Confusão - MLP Bom Ajuste")
plot_learning_curve(mlp_goodfit.train_losses, mlp_goodfit.val_losses, title="Curva de Aprendizado - MLP Bom Ajuste", loss_label="BCE")


# Caso de Overfitting (MLP Superdimensionado)
print("\n--- MLP: Caso de Overfitting ---")
neurons_over_fit = 100
epochs_over_fit = 1000 # Aumentado
lr_over_fit = 0.05     # Mesmo lr do bom
mlp_overfit = MLP(n_input_biased=n_features_biased_mlp, n_hidden=neurons_over_fit, n_output=n_output_mlp, 
                  learning_rate=lr_over_fit, epochs=epochs_over_fit) 
print(f"Topologia MLP Overfitting: 1 camada oculta com {neurons_over_fit} neurônios.")
mlp_overfit.fit(X_train_mlp_demo_biased, y_train_mlp_demo_col, X_val_mlp_demo_biased, y_val_mlp_demo_col)

y_pred_proba_over = mlp_overfit.predict_proba(X_val_mlp_demo_biased)
tp_o, tn_o, fp_o, fn_o, acc_o, sens_o, spec_o = calculate_metrics(y_val_mlp_demo, y_pred_proba_over)
cm_o = confusion_matrix_manual(y_val_mlp_demo, y_pred_proba_over)

print("\nDesempenho MLP Overfitting (Validação):")
print(f"  Acurácia: {acc_o:.8f}")
print(f"  Sensibilidade: {sens_o:.8f}")
print(f"  Especificidade: {spec_o:.8f}")
print("  Matriz de Confusão:")
print(cm_o)
if sum(sum(cm_o)) > 0 : plot_confusion_matrix(cm_o, title="Matriz de Confusão - MLP Overfitting")
plot_learning_curve(mlp_overfit.train_losses, mlp_overfit.val_losses, title="Curva de Aprendizado - MLP Overfitting", loss_label="BCE")
print("-" * 30)

# --- 5. Validação por Monte Carlo ---
print("\n5. Validação por Monte Carlo (R=250)")
R = 30 # REDUZIDO PARA TESTES RÁPIDOS. Mude para 250 para a execução final.
print(f"AVISO: Número de rodadas Monte Carlo (R) está configurado para {R} para teste rápido.")
test_fraction = 0.2

# Hiperparâmetros para Monte Carlo (usar os do "bom ajuste" ou otimizados)
lr_sp_mc = 0.01
epochs_sp_mc = 600

lr_mlp_mc = lr_good_fit
epochs_mlp_mc = epochs_good_fit
hidden_neurons_mlp_mc = neurons_good_fit

results_sp = {'accuracy': [], 'sensitivity': [], 'specificity': [], 'train_losses_all_runs': [], 'val_losses_all_runs': []} # val_losses aqui é conceitual
results_mlp = {'accuracy': [], 'sensitivity': [], 'specificity': [], 'train_losses_all_runs': [], 'val_losses_all_runs': []}

run_data_sp = []
run_data_mlp = []

for r_idx in range(R):
    current_mc_seed = r_idx 
    
    print(f"Rodada Monte Carlo: {r_idx+1}/{R} (semente da partição: {current_mc_seed})")
    
    X_train, X_test, y_train, y_test = train_test_split_manual(X_scaled, y_orig, test_size=test_fraction, random_state=current_mc_seed)
    
    # Debug: Verificar balanceamento em cada rodada
    if r_idx < 2: # Só para as primeiras rodadas
        print(f"  MC Rodada {r_idx+1} - y_train unique: {np.unique(y_train, return_counts=True)}")
        print(f"  MC Rodada {r_idx+1} - y_test unique: {np.unique(y_test, return_counts=True)}")
        if len(np.unique(y_test)) < 2:
            print(f"  AVISO: y_test na rodada {r_idx+1} tem menos de 2 classes! Isso afetará as métricas.")


    X_train_b = add_bias(X_train)
    X_test_b = add_bias(X_test)
    y_train_col = y_train.reshape(-1, 1)
    y_test_col = y_test.reshape(-1, 1) # Para MLP, embora calculate_metrics use y_test plano

    # Perceptron Simples
    sp = SimplePerceptron(learning_rate=lr_sp_mc, n_epochs=epochs_sp_mc)
    sp.fit(X_train_b, y_train)
    y_pred_proba_sp_mc = sp.predict_proba(X_test_b)
    _, _, _, _, acc_sp_mc, sens_sp_mc, spec_sp_mc = calculate_metrics(y_test, y_pred_proba_sp_mc)
    
    results_sp['accuracy'].append(acc_sp_mc)
    results_sp['sensitivity'].append(sens_sp_mc)
    results_sp['specificity'].append(spec_sp_mc)
    # Para SP, errors_ (atualizações) é a "curva de aprendizado" de treino.
    # Não há "val_loss" no mesmo sentido que MLP.
    run_data_sp.append({'X_train_b': X_train_b, 'y_train': y_train, 
                        'X_test_b': X_test_b, 'y_test': y_test, 
                        'acc': acc_sp_mc, 'model_errors': sp.errors_ })


    # MLP
    mlp = MLP(n_input_biased=X_train_b.shape[1], n_hidden=hidden_neurons_mlp_mc, n_output=1, 
              learning_rate=lr_mlp_mc, epochs=epochs_mlp_mc)
    # Fit SEM X_val_biased, y_val_col aqui para não imprimir logs de época no MC
    # A curva de aprendizado para o item 6 será gerada re-treinando no split específico
    mlp.fit(X_train_b, y_train_col) # Sem X_val, y_val para reduzir output no MC
    
    y_pred_proba_mlp_mc = mlp.predict_proba(X_test_b)
    _, _, _, _, acc_mlp_mc, sens_mlp_mc, spec_mlp_mc = calculate_metrics(y_test, y_pred_proba_mlp_mc)
    
    results_mlp['accuracy'].append(acc_mlp_mc)
    results_mlp['sensitivity'].append(sens_mlp_mc)
    results_mlp['specificity'].append(spec_mlp_mc)
    # Para o item 6, vamos precisar treinar novamente no split específico para obter as curvas com validação
    # Então, aqui, podemos só salvar o necessário para refazer o predict, ou salvar os pesos.
    # Por simplicidade, salvaremos os dados do split e re-treinaremos.
    run_data_mlp.append({'X_train_b': X_train_b, 'y_train_col': y_train_col, 
                         'X_test_b': X_test_b, 'y_test_col': y_test_col,
                         'acc': acc_mlp_mc}) # Guardamos y_test_col para consistência

print("Validação Monte Carlo concluída.")
print("-" * 30)

# --- 6. Análise das Rodadas de Maior e Menor Acurácia ---
print("\n6. Análise das Rodadas de Maior e Menor Acurácia")

def analyze_best_worst_run_mc(model_name, results_acc_list, run_data_list, 
                           is_mlp=False, sp_lr=0.01, sp_epochs=600,
                           mlp_lr=0.05, mlp_epochs=500, mlp_hidden=10): # Passar params
    print(f"\n--- {model_name} (Monte Carlo) ---")
    if not results_acc_list:
        print("Nenhum resultado para analisar.")
        return

    accuracies = np.array(results_acc_list)
    idx_max_acc = np.argmax(accuracies)
    idx_min_acc = np.argmin(accuracies)

    print(f"Melhor Acurácia: {accuracies[idx_max_acc]:.8f} (Rodada MC original {idx_max_acc+1})")
    print(f"Pior Acurácia: {accuracies[idx_min_acc]:.8f} (Rodada MC original {idx_min_acc+1})")

    for case, idx in [("Melhor Acurácia", idx_max_acc), ("Pior Acurácia", idx_min_acc)]:
        print(f"\nAnalisando {model_name} - {case}:")
        current_run_data = run_data_list[idx]
        
        X_train_case_b = current_run_data['X_train_b']
        X_test_case_b = current_run_data['X_test_b']
        
        if is_mlp:
            y_train_case_col = current_run_data['y_train_col']
            y_test_case_col = current_run_data['y_test_col'] # y_test já está como coluna aqui
            y_test_case_flat = y_test_case_col.flatten()

            # Re-treinar MLP no split específico para obter curvas de aprendizado com validação
            temp_mlp = MLP(n_input_biased=X_train_case_b.shape[1], n_hidden=mlp_hidden, n_output=1, 
                           learning_rate=mlp_lr, epochs=mlp_epochs)
            print(f"Re-treinando MLP para {case} (curvas de aprendizado)...")
            temp_mlp.fit(X_train_case_b, y_train_case_col, X_val_biased=X_test_case_b, y_val_col=y_test_case_col)
            
            y_pred_proba_case = temp_mlp.predict_proba(X_test_case_b)
            plot_learning_curve(temp_mlp.train_losses, temp_mlp.val_losses, 
                                title=f"Curva de Aprendizado - {model_name} ({case})", loss_label="BCE")
        else: # Perceptron Simples
            y_train_case = current_run_data['y_train']
            y_test_case_flat = current_run_data['y_test']

            # Re-treinar SP para obter predições e curva de erros
            temp_sp = SimplePerceptron(learning_rate=sp_lr, n_epochs=sp_epochs)
            temp_sp.fit(X_train_case_b, y_train_case)
            y_pred_proba_case = temp_sp.predict_proba(X_test_case_b)
            
            plt.figure(figsize=(8,6))
            plt.plot(range(1, len(temp_sp.errors_) + 1), temp_sp.errors_, marker='o')
            plt.xlabel('Épocas')
            plt.ylabel('Número de Atualizações de Peso (Treino)')
            plt.title(f'Curva de Aprendizado (Erros de Treino) - {model_name} ({case})')
            plt.grid(True)
            plt.savefig(f'imgs/spiral3d_perceptron_learning_curve_{case}.png', dpi=300)
            plt.show()

        cm_case = confusion_matrix_manual(y_test_case_flat, y_pred_proba_case)
        if sum(sum(cm_case)) > 0 : plot_confusion_matrix(cm_case, title=f"Matriz de Confusão - {model_name} ({case})")
        else: print("Matriz de confusão vazia (sem dados de teste ou todas as predições foram 0?).")


# Análise para Perceptron Simples (Monte Carlo)
analyze_best_worst_run_mc("Perceptron Simples", results_sp['accuracy'], run_data_sp, is_mlp=False,
                       sp_lr=lr_sp_mc, sp_epochs=epochs_sp_mc)

# Análise para MLP (Monte Carlo)
analyze_best_worst_run_mc("MLP", results_mlp['accuracy'], run_data_mlp, is_mlp=True,
                       mlp_lr=lr_mlp_mc, mlp_epochs=epochs_mlp_mc, mlp_hidden=hidden_neurons_mlp_mc)
print("-" * 30)


# --- 7. Estatísticas Finais e Discussão ---
print("\n7. Estatísticas Finais das R=250 Rodadas e Discussão")

def print_stats_table(metric_name, sp_metrics_list, mlp_metrics_list):
    print(f"\n--- Tabela para: {metric_name} ---")
    header = f"{'Modelo':<20} | {'Média':<10} | {'Desvio Padrão':<15} | {'Mínimo':<10} | {'Máximo':<10}"
    print(header)
    print("-" * len(header))
    
    models_data = {
        "Perceptron Simples": sp_metrics_list,
        "MLP": mlp_metrics_list
    }
    
    for model_name, metrics_list in models_data.items():
        if not metrics_list: # Lista vazia
            print(f"{model_name:<20} | {'N/A':<10} | {'N/A':<15} | {'N/A':<10} | {'N/A':<10}")
            continue
        
        mean_val = np.mean(metrics_list)
        std_val = np.std(metrics_list)
        min_val = np.min(metrics_list)
        max_val = np.max(metrics_list)
        print(f"{model_name:<20} | {mean_val:<10.4f} | {std_val:<15.4f} | {min_val:<10.4f} | {max_val:<10.4f}")


# Acurácia
print_stats_table("Acurácia", results_sp['accuracy'], results_mlp['accuracy'])
# Sensibilidade
print_stats_table("Sensibilidade", results_sp['sensitivity'], results_mlp['sensitivity'])
# Especificidade
print_stats_table("Especificidade", results_sp['specificity'], results_mlp['specificity'])


if results_sp['accuracy'] and results_mlp['accuracy']: # Só plota se houver dados
    metric_names_plot = ['Acurácia', 'Sensibilidade', 'Especificidade']
    data_sp_plot = [results_sp['accuracy'], results_sp['sensitivity'], results_sp['specificity']]
    data_mlp_plot = [results_mlp['accuracy'], results_mlp['sensitivity'], results_mlp['specificity']]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False) # sharey=False pode ser melhor
    fig.suptitle('Comparação de Desempenho dos Modelos (Boxplots de R Rodadas)', fontsize=16)

    for i, metric_name_p in enumerate(metric_names_plot):
        # Filtrar listas vazias antes de passar para boxplot
        plot_data = []
        labels = []
        if data_sp_plot[i]:
            plot_data.append(data_sp_plot[i])
            labels.append('Perceptron')
        if data_mlp_plot[i]:
            plot_data.append(data_mlp_plot[i])
            labels.append('MLP')
        
        if plot_data: # Se houver dados para plotar
            axes[i].boxplot(plot_data, labels=labels)
            axes[i].set_title(metric_name_p)
            axes[i].set_ylabel('Valor da Métrica')
            axes[i].grid(True, linestyle='--', alpha=0.7)
        else:
            axes[i].text(0.5, 0.5, 'Sem dados para plotar', horizontalalignment='center', verticalalignment='center')
            axes[i].set_title(metric_name_p)


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('imgs/spiral3d_boxplots.png', dpi=300)
    plt.show()
else:
    print("Não foi possível gerar boxplots (sem resultados suficientes do Monte Carlo).")

print("\nDiscussão dos Resultados (será mais significativa após rodar com R=250 e hiperparâmetros ajustados):")
# (Manter a discussão original, mas ela dependerá dos novos resultados)
print("O conjunto de dados 'Spiral3d' é conhecido por ser não linearmente separável...")
# ... (resto da discussão)

print("-" * 30)
print("FIM DA ANÁLISE")