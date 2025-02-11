class KNearestNeighbors:
    def __init__(self, neighbors=5):
        # Set number of neighbors for classification
        self.neighbors = neighbors
    
    def fit(self, X_train, Y_train):
        # Store training data (features and labels)
        self.X_train = X_train
        self.Y_train = Y_train
    
    def classify(self, X_test):
        # Predict labels for test data
        return [self._find_class(x) for x in X_test]
    
    def _find_class(self, x):
        # Calculate distances from test point to all training points
        distances = [self._euclidean_distance(x, train_point) for train_point in self.X_train]
        
        # Find nearest neighbors and their labels
        nearest_indices = self._sort_indices(distances)[:self.neighbors]
        nearest_labels = [self.Y_train[i] for i in nearest_indices]
        
        # Return most common label among neighbors
        return self._most_common(nearest_labels)
    
    def _euclidean_distance(self, x1, x2):
        # Compute Euclidean distance between two points
        return sum((xi - xj) ** 2 for xi, xj in zip(x1, x2)) ** 0.5
    
    def _sort_indices(self, distances):
        # Sort distances and return indices of nearest points
        return sorted(range(len(distances)), key=lambda i: distances[i])
    
    def _most_common(self, labels):
        # Return the most frequent label
        count = {}
        for label in labels:
            count[label] = count.get(label, 0) + 1
        return max(count, key=count.get)

# Example usage
if __name__ == "__main__":
    # Training data
    X_train = [[3, 6], [2, 7], [8, 3], [6, 5], [1, 2], [9, 8], [7, 4], [5, 9]]
    Y_train = ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B']
    
    # Test data
    X_test = [[4, 5], [7, 2]]  # Points to classify
    
    # Train and predict
    knn = KNearestNeighbors(neighbors=4)
    knn.fit(X_train, Y_train)
    results = knn.classify(X_test)
    
    # Output results
    print("Predicted Labels:", results)
