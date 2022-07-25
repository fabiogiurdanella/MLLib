# Try to build a classifier for the MNIST dataset that achieves over 97% accuracy
# on the test set. Hint: the KNeighborsClassifier works quite well for this task;
# you just need to find good hyperparameter values (try a grid search on the
# weights and n_neighbors hyperparameters).

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from dataset import Dataset

# Estrazione dati
dataset = Dataset()
X_train = dataset.get_train_set()
y_train = dataset.get_train_labels()
X_test = dataset.get_test_set()
y_test = dataset.get_test_labels()



# Creazione classificatore
classificator = KNeighborsClassifier(n_neighbors=3, weights="distance")
classificator.fit(X_train, y_train)

# # Calcolo precisione
y_results = classificator.predict(X_test)

precision = precision_score(y_test, y_results, average="micro") # Precision = TP / (TP + FP)
recall = recall_score(y_test, y_results, average="micro") # Recall = TP / (TP + FN)

print(precision, recall)

