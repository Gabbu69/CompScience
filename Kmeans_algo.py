import numpy as np

class KMeans:
    def __init__(self, k=5, maxIters=200):
        self.k = k
        self.maxIters = maxIters
        self.centroids = None
    
    def fit(self, X):
        X = np.array(X)
        np.random.seed(42)  # changed the seed
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]
        
        for _ in range(self.maxIters):
            clusters = self._assignClusters(X)
            newCentroids = self._computeCentroids(X, clusters)
            
            if np.all(self.centroids == newCentroids):
                break
            
            self.centroids = newCentroids
        
    def predict(self, X):
        return self._assignClusters(np.array(X))
    
    def _assignClusters(self, X):
        distances = np.array([[np.linalg.norm(x - centroid) for centroid in self.centroids] for x in X])
        return np.argmin(distances, axis=1)
    
    def _computeCentroids(self, X, clusters):
        return np.array([X[clusters == i].mean(axis=0) for i in range(self.k)])

# sample usage:
if __name__ == "__main__":
    X = np.array([[4, 3], [5, 6], [2, 7], [15, 13], [16, 14], [14, 15], [6, 7], [10, 9], [9, 10], [15, 10]])
    
    kmeans = KMeans(k=5)
    kmeans.fit(X)
    clusters = kmeans.predict(X)
    
    print("Cluster assignments:", clusters)
    print("Centroids:", kmeans.centroids)
