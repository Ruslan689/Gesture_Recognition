import numpy as np
from cvxopt import matrix, solvers

class ManualSVM:
    def __init__(self, C=1.0, gamma=0.1):
        self.C = C
        self.gamma = gamma
        self.models = {}

    def _kernel(self, X1, X2):
        pairwise_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1) - 2 * X1.dot(X2.T)
        return np.exp(-self.gamma * pairwise_dists)

    def _fit_binary(self, X, y):
        n_samples = X.shape[0]
        K = self._kernel(X, X)
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        A = matrix(y.reshape(1, -1).astype('double'))
        b = matrix(0.0)

        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x']).flatten()
        sv = alphas > 1e-5
        bias = np.mean(y[sv] - (alphas[sv] * y[sv]).dot(K[sv][:, sv]))
        return alphas[sv], X[sv], y[sv], bias

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            y_binary = np.where(y == cls, 1, -1)
            alphas, sv_X, sv_y, b = self._fit_binary(X, y_binary)
            self.models[cls] = (alphas, sv_X, sv_y, b)

    def predict(self, X):
        sv_all = []
        class_boundaries = []
        for (alphas, sv_X, sv_y, b) in self.models.values():
            sv_all.append(sv_X)
            class_boundaries.append(len(sv_X))
        sv_all = np.vstack(sv_all)

        K_test = self._kernel(sv_all, X)
        votes = np.zeros((X.shape[0], len(self.classes)))

        start = 0
        for i, (cls, (alphas, sv_X, sv_y, b)) in enumerate(self.models.items()):
            end = start + len(sv_X)
            k = K_test[start:end]
            votes[:, i] = (alphas * sv_y).dot(k) + b
            start = end

        return self.classes[np.argmax(votes, axis=1)]
