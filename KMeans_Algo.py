class KMeans:
    def __init__(self, k=5, maxIters=200):
        self.k = k
        self.maxIters = maxIters
        self.centroids = None
    
    def fit(self, X):
        # Convert data to numpy array
        X = np.array(X)
        
        # Randomly initialize centroids
        np.random.seed(42)
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]
        
        for _ in range(self.maxIters):
            # Assign clusters based on closest centroid
            clusters = self._assignClusters(X)
            
            # Recalculate centroids
            newCentroids = self._computeCentroids(X, clusters)
            
            # If centroids don't change, stop
            if np.all(self.centroids == newCentroids):
                break
            
            self.centroids = newCentroids
        
    def predict(self, X):
        # Assign clusters to new data points
        return self._assignClusters(np.array(X))
    
    def _assignClusters(self, X):
        # Calculate distances from each point to each centroid
        distances = np.array([[np.linalg.norm(x - centroid) for centroid in self.centroids] for x in X])
        return np.argmin(distances, axis=1)
    
    def _computeCentroids(self, X, clusters):
        # Compute new centroids based on mean of points in each cluster
        return np.array([X[clusters == i].mean(axis=0) for i in range(self.k)])

# sample usage:
if __name__ == "__main__":
    X = [[4, 3], [5, 6], [2, 7], [15, 13], [16, 14], [14, 15], [6, 7], [10, 9], [9, 10], [15, 10]]
    
    kmeans = KMeans(k=5)
    kmeans.fit(X)
    clusters = kmeans.predict(X)
    
    print("Cluster assignments:", clusters)
    print("Centroids:", kmeans.centroids)
