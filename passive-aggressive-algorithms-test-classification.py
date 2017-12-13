import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Set random seed
np.random.seed(1000)

nb_samples = 5000
nb_features = 4

# Create our dataset
X, Y = make_classification(n_samples=nb_samples,
                           n_features=nb_features,
                           n_informative=nb_features - 2,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           n_clusters_per_class=2)

# Split our dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=1000)

# Perform logoistic regression
lr = LogisticREgression()
lr.fit(X_train, Y_train)
print('Score: {}'.format(lr.score(X_test, Y_test)))

# Set the y=0 labels to -1
Y_train[Y_train==0] = -1
Y_test[Y_test==0] = -1

c = 0.01
w = np.zeros((nb_features, 1))

# Implement a Passive Aggressive Classification
for i in range(X_train.shape[0]):
    xi = X_train[i].reshape((nb_features, 1))
    
    loss = max(0, 1 - (Y_train[i] * np.dot(w.T, xi)))
    tau = loss / (np.power(np.linalg.norm(xi, ord=2), 2) + (1 / (2*C)))
    
    coeff = tau * Y_train[i]
    w += coeff * xi
    
# Compute accuracy
Y_pred = np.sign(np.dot(w.T, X_test.T))
c = np.count_nonzero(Y_pred - Y_test)

print('PA accuracy: {}'.format(1 - float(c) / X_test.shape[0]))