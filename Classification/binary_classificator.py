from sklearn.linear_model import SGDClassifier
from dataset import Dataset

def binary_classificator(X_train, y_train):
    """
            For each instance, it computes a score based on a decision function, 
            and if that score is greater than a threshold, it assigns the instance to the positive
            class, or else it assigns it to the negative class.
    """
    # Creating a classifier that is able to classify if a number is 8 or not.
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train)
    return sgd_clf

def confusion_matrix_validation(sgd_clf: SGDClassifier, X_train, y_train):
    """
        The general idea is to count the number of times instances of class A are
        classified as class B. A perfect classifier would have only true positives and true
        negatives,
    """
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix
    y_train_previsti = cross_val_predict(sgd_clf, X_train, y_train, cv=3) # Restituisce un array di boolean per ogni X_train di risultati previsti
    # Matrix 2 x 2
    # [[True Positive, False Positive] 
    # [False Negative, True Negative]]
    
    matrix = confusion_matrix(y_train, y_train_previsti) 
    precision = matrix[0][0] / (matrix[0][0] + matrix[0][1]) # P = TP / (TP + FP)
    return precision
    

dataset = Dataset()
X_train = dataset.get_train_set()
print(X_train)

y_train = dataset.get_train_labels()

X_test = dataset.get_test_set()
y_test = dataset.get_test_labels()

y_train_8 = (y_train == 8)

classificator = binary_classificator(X_train, y_train_8)
validation = confusion_matrix_validation(classificator, X_train, y_train_8)
print(validation)

result = classificator.predict(X_test.iloc[5].values.reshape(1, -1))