#!/usr/bin/env python
# coding: utf-8

# # Momento de Retroalimentación: 
# ## Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework.
# ## (Portafolio Implementación)
# ### Alan Contreras Prieto - A01749667

# In[14]:


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

from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# # Etapa anterior

# In[15]:


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


# In[16]:


# Dividir datos en entrenamiento y  prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[17]:


tree_clf = tree.DecisionTreeClassifier()

folds = KFold(n_splits=5) #creación de los folds 

dicHiper = {} 
dicHiper["criterion"] = ['gini', 'entropy',"log_loss"] 
dicHiper["splitter"] = ['best', 'random'] 
dicHiper["max_depth"] = [5,10,15,20,25,30] 
dicHiper["min_samples_split"] = [50,100,250,500]

hpSearch = GridSearchCV(tree_clf,dicHiper,n_jobs=-1,scoring="accuracy",cv=folds,verbose=3) 

result = hpSearch.fit(X,y) 
print(f"Mejor score {result.best_score_}") 
print(f"Mejores hiperparámetros: {result.best_params_}\n\n\n") 


# In[18]:


tree_clf = result.best_estimator_
tree_clf.fit(X_train,y_train)
y_pred = tree_clf.predict(X_test)


# In[19]:


# Reporte de métricas
print("\nReporte de Clasificación:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

df1 = pd.DataFrame(columns=["0","1"], index= ["0","1"], data= cm )

f,ax = plt.subplots(figsize=(8,8))

sns.heatmap(df1, annot=True,cmap="Reds", fmt= '.0f',
            ax=ax,linewidths = 5, cbar = False,annot_kws={"size": 14})
plt.xlabel("Predicción")
plt.ylabel("REal")
plt.title("Árbol de decisión con datos de prueba")
plt.show()


# # Dataset

# In[20]:


data = sklearn.datasets.make_classification(
    n_samples=50000, 
    n_features=10,
    n_classes=5,

    n_informative=5
)

X = data[0]
y = data[1]
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'])
df['y'] = y
df["y"].value_counts()


# In[21]:


# Dividir datos en entrenamiento y  prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# # Modelos

# ## Arboles de decisión

# In[22]:


tree_clf = RandomForestClassifier()

folds = KFold(n_splits=5) #creación de los folds

dicHiper = {} 
dicHiper["n_estimators"] = [10, 50, 100]
dicHiper["criterion"] = ['gini', 'entropy',"log_loss"] 
dicHiper["max_depth"] = [10,20,30, None] 
dicHiper["min_samples_split"] = [50,100,250,500]

hpSearch = GridSearchCV(tree_clf,dicHiper,n_jobs=-1,scoring="accuracy",cv=folds,verbose=3)

result = hpSearch.fit(X_train,y_train)
print(f"Mejor score {result.best_score_}") 
print(f"Mejores hiperparámetros: {result.best_params_}\n\n\n")


# ### Entrenamiento

# In[23]:


tree_clf = result.best_estimator_
tree_clf.fit(X_train,y_train)
y_pred = tree_clf.predict(X_test)


# In[24]:


# Reporte de métricas
print("\nReporte de Clasificación:\n")
print(classification_report(y_test, y_pred, target_names=["Clase 0", "Clase 1", "Clase 2", "Clase 3", "Clase 4"]))



cm = confusion_matrix(y_test, y_pred)

df1 = pd.DataFrame(columns=["0","1","2","3","4"], index= ["0","1","2","3","4"], data= cm )

f,ax = plt.subplots(figsize=(8,8))

sns.heatmap(df1, annot=True,cmap="Reds", fmt= '.0f',
            ax=ax,linewidths = 5, cbar = False,annot_kws={"size": 14})
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Random Forest con datos de prueba ", size = 10)
plt.show()


# # Resultados

# In[ ]:





# # Importar a .py

# In[25]:


#!jupyter nbconvert --to script Practica2.ipynb

