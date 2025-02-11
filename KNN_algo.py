import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self, neighbors=5):
        self.neighbors = neighbors
    
    def fit(self, X_train, Y_train):
        self.X_train = np.array(X_train)
        self.Y_train = np.array(Y_train)
    
    def classify(self, X_test):
        return np.array([self._find_class(x) for x in X_test])
    
    def _find_class(self, x):
        distances = np.sqrt(((self.X_train - x) ** 2).sum(axis=1))
        nearest_indices = np.argsort(distances)[:self.neighbors]
        nearest_labels = [self.Y_train[i] for i in nearest_indices]
        return Counter(nearest_labels).most_common(1)[0][0]

# Example usage
if __name__ == "__main__":
    X_train = [[3, 6], [2, 7], [8, 3], [6, 5], [1, 2], [9, 8], [7, 4], [5, 9]]
    Y_train = ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B']
    
    X_test = [[4, 5], [7, 2]]  # Points to classify
    
    knn = KNearestNeighbors(neighbors=4)
    knn.fit(X_train, Y_train)
    results = knn.classify(X_test)
    print("Predicted Labels:", results)
    