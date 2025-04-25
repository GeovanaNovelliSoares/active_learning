import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel='linear', probability=True)

def uncertainty_sampling(X_train, y_train, model, n_samples=10):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_train)
    uncertainty = np.max(probs, axis=1)  
    query_idx = np.argsort(uncertainty)[:n_samples] 
    
    return query_idx

accuracies = []

n_iterations = 10
for i in range(n_iterations):
    print(f"Iteração {i + 1}:")
    query_idx = uncertainty_sampling(X_train, y_train, svm, n_samples=10)
    X_query, y_query = X_train[query_idx], y_train[query_idx]
    X_train = np.concatenate([X_train, X_query])
    y_train = np.concatenate([y_train, y_query])
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    disp.plot(cmap=plt.cm.Blues, ax=ax[0])
    ax[0].set_title(f'Matriz de Confusão - Iteração {i + 1}')

    ax[1].plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-', color='b')
    ax[1].set_title('Evolução da Acurácia do Modelo')
    ax[1].set_xlabel('Iteração')
    ax[1].set_ylabel('Acurácia')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()
    print(f"Acurácia do modelo: {accuracy * 100:.2f}%\n")

pca = PCA(n_components=2)
X_2D = pca.fit_transform(X_train)
plt.figure(figsize=(8, 6))
plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y_train, cmap='viridis', marker='o', alpha=0.6)
plt.title('Visualização das Amostras Selecionadas no Plano 2D (PCA)')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.colorbar(label='Classe')
plt.show()
