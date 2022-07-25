from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_curve
import numpy as np
from dataset import Dataset
import matplotlib.pyplot as plt

def binary_classificator(X_train, y_train):
    """
            For each instance, it computes a score based on a decision function, # https://it.wikipedia.org/wiki/Discesa_stocastica_del_gradiente
            and if that score is greater than a threshold, it assigns the instance to the positive
            class, or else it assigns it to the negative class.
    """
    # Creating a classifier that is able to classify if a number is 8 or not.
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train)
    return sgd_clf

# def predict_by_threshold(classificator: SGDClassifier, X_train, y_train, precision_threshold: float):
#     y_scores = cross_val_predict(classificator, X_train, y_train, cv=3, method="decision_function") 
#     # precision = TP / (TP + FP) --> Abilità del classificatore nel restituire il valore corretto
#     # recall = TP / (TP + FN) --> Abilità del classificatore di restituitre il valore true correttamente
#     precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
#     threshold = thresholds[np.argmax(precisions >= precision_threshold)]
#     y_train_previsti = y_scores >= threshold # Nuovo cross_val_predict basato sul nuovo threesold

    
    
def show_confusion_matrix(sgd_clf: SGDClassifier, X_train, y_train):
    """
        The general idea is to count the number of times instances of class A are
        classified as class B. A perfect classifier would have only true positives and true
        negatives
        For example, to know the number of times the classifier confused
        images of 5s with 3s, you would look in the 5th row and 3rd column of the confusion
        matrix
    """
    y_train_previsti = cross_val_predict(sgd_clf, X_train, y_train, cv=3) # Restituisce un array di boolean per ogni X_train di risultati previsti
    # Matrix 2 x 2
    # [[True Positive, False Positive] 
    # [False Negative, True Negative]]
    matrix = confusion_matrix(y_train, y_train_previsti) 
    plt.matshow(matrix, cmap=plt.cm.gray)
    plt.show()

def show_error_with_confusion_matrix(sgd_clf: SGDClassifier, X_train, y_train):
    y_train_previsti = cross_val_predict(sgd_clf, X_train, y_train, cv=3) # Restituisce un array di boolean per ogni X_train di risultati previsti
    matrix = confusion_matrix(y_train, y_train_previsti) 
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_conf_matrix = matrix / row_sums # Divido ogni valore della matrice per il numero di elementi della riga
    np.fill_diagonal(norm_conf_matrix, 0)
    plt.matshow(norm_conf_matrix, cmap=plt.cm.gray)
    plt.show()
    
    

# dataset = Dataset()

# X_train = dataset.get_train_set()
# y_train = dataset.get_train_labels()

# X_test = dataset.get_test_set()
# y_test = dataset.get_test_labels()

# y_train_8 = (y_train == 8)

# # classificator = binary_classificator(X_train, y_train_8) # Classificatore che classifica se un numero è 8 o no

# classificator_multiple = binary_classificator(X_train, y_train) # In questo caso, under the hood, scikit crea 10 classificatori, uno per ogni label (0,1,2...9). Per ogni predict, si sceglie quello che ha un risultato massimo migliore e viene restituito
# show_confusion_matrix(classificator_multiple, X_train, y_train)
# show_error_with_confusion_matrix(classificator_multiple, X_train, y_train)

# result = classificator_multiple.predict(X_train.iloc[145].values.reshape(1, -1))
# print(result)