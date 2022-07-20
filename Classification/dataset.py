import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from joblib import Memory

# some_digit = X.iloc[20] # 10th img in the dataset
# some_digit_image = some_digit.values.reshape(28, 28)
# plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()

# Creazione training set e test set.
# Per questo dataset, i dati sono già separati in training set e test set e già mischiati
# Bisogna di solito MISCHIARE I DATASET

class Dataset:
    def __init__(self):
        memory = Memory('./tmp')
        fetch_openml_cached = memory.cache(fetch_openml)
        mnist = fetch_openml_cached('mnist_784', version=1)

        self.X = mnist.data # Matrix of features --> Shape(70000, 784 (28*28)) --> 70000 img, 28*28 pixels
        y = mnist.target # Vector of labels --> Shape(70000,)
        self.y = y.astype(int) # Vector of labels --> Shape(70000,)


    def get_train_set(self):
        # Returning the first 60,000 rows of the X matrix.
        return self.X[0:60000]

    def get_test_set(self):
        # Returning the last 10,000 rows of the X matrix.
        return self.X[60000:]

    def get_train_labels(self):
        # Returning the first 60,000 rows of the y vector.
        return self.y[0:60000]

    def get_test_labels(self):
        # Returning the last 10,000 rows of the y vector.
        return self.y[60000:]
