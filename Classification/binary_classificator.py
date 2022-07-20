from dataset import Dataset

def binary_classificator(X_train, y_train):
    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train)
    return sgd_clf


dataset = Dataset()
X_train = dataset.get_train_set()
y_train = dataset.get_train_labels()

X_test = dataset.get_test_set()
y_test = dataset.get_test_labels()

result = binary_classificator(X_train, (y_train == 8)).predict(X_test.iloc[78].values.reshape(1, -1))
print(result)