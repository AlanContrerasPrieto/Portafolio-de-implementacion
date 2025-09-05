# Momento de Retroalimentación: 
# Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. 
# (Portafolio Implementación)
# Alan Contreras Prieto - A01749667


# Librerias
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[142]:


data = sklearn.datasets.make_classification(
    n_samples=10000, 
    n_features=5,  
    n_classes=2
)

X = data[0]
y = data[1]
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])
df['y'] = y
df["y"].value_counts()


# In[143]:


df.describe()


# In[144]:


# Dividir datos en entrenamiento y  prueba
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[145]:


# Red
class RedNeuronal:
    # Inicialización 
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.weightsΔ = []
        self.biasesΔ = []

        # Inicialización de pesos y biases aleatorios
        for i in range(len(layers) - 1):
            # Pesos: matriz de (neuronas capa actual, neuronas capa siguiente)
            w = np.random.randn(layers[i], layers[i+1])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        # Initialize each Δw and Δb to zero.
        for i in range(len(self.layers) - 1):
            wΔ = np.zeros((self.layers[i], self.layers[i+1]))
            bΔ = np.zeros((1, self.layers[i+1]))
            self.weightsΔ.append(wΔ)
            self.biasesΔ.append(bΔ)

    # Mostrar pesos
    def show_weights(self):
        for i in range(len(self.layers)-1):
            print(f"Capa {i+1}-{i+2}")
            print("Weights: ", self.weights[i],"Bias: ",self.biases[i])

    #Función de activación sigmoide
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Feedforward
    def predict(self, X):
        a = X
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = self.sigmoid(z)
        return a

    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            # Feedforward
            a = self.predict(X)

            # loss (MSE)
            loss = np.mean((a - y) ** 2)

            # Backward pass (gradient descent)
            self.backward(X, y, a, learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def backward(self, X, y, a, learning_rate):
        # Paso 1: forward con almacenamiento de activaciones y z
        activations = [X]
        zs = []  # valores antes de activación

        a_tmp = X
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a_tmp, w) + b
            zs.append(z)
            a_tmp = self.sigmoid(z)
            activations.append(a_tmp)

        # Paso 2: error de salida
        delta = (activations[-1] - y) * (activations[-1] * (1 - activations[-1]))

        # Gradientes de última capa
        nabla_w = [np.dot(activations[-2].T, delta)]
        nabla_b = [np.sum(delta, axis=0, keepdims=True)]

        # Paso 3: backpropagation en capas ocultas
        for l in range(2, len(self.layers)):
            z = zs[-l]
            sp = activations[-l] * (1 - activations[-l])  # derivada de sigmoide
            delta = np.dot(delta, self.weights[-l+1].T) * sp
            nabla_w.insert(0, np.dot(activations[-l-1].T, delta))
            nabla_b.insert(0, np.sum(delta, axis=0, keepdims=True))

        # Paso 4: actualizar pesos y biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * nabla_w[i]
            self.biases[i] -= learning_rate * nabla_b[i]


# In[146]:


capas = [5,       # Capa de entrada
       3,5,3,   # Capas ocultas
       1]       # Capa de salida
red = RedNeuronal(capas)
print("Pesos iniciales (antes de entrenamiento):")
red.show_weights()


# In[147]:


y_pred = (red.predict(x_train) >= 0.5).astype(int)  # predicciones binarizadas

# Reporte de métricas
print("\nReporte de Clasificación:\n")
print(classification_report(y_train, y_pred, target_names=["Clase 0", "Clase 1"]))

# Crear matriz de confusión
cm = confusion_matrix(y_train, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión: antes de entrenar el modelo")
plt.show()


# In[148]:


red.fit(x_train, y_train.reshape(-1, 1), epochs=1001, learning_rate=0.01)


# In[149]:


print("Pesos finales (despues de entrenamiento):")
red.show_weights()


# In[150]:


y_pred = (red.predict(x_train) >= 0.5).astype(int)  # predicciones binarizadas

# Reporte de métricas
print("\nReporte de Clasificación:\n")
print(classification_report(y_train, y_pred, target_names=["Clase 0", "Clase 1"]))

# Crear matriz de confusión
cm = confusion_matrix(y_train, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión: despues de entrenar el modelo")
plt.show()


# In[151]:


y_pred = (red.predict(x_test) >= 0.5).astype(int)  # predicciones binarizadas

# Reporte de métricas
print("\nReporte de Clasificación:\n")
print(classification_report(y_test, y_pred, target_names=["Clase 0", "Clase 1"]))

# Crear matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión: datos de prueba")
plt.show()