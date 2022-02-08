import numpy as np


def relu(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X


def softmax(X):
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]

    return X

def inplace_relu_derivative(Z, delta):
    delta[Z == 0] = 0

def log_loss(y_true, y_prob):
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return - np.multiply(y_true, np.log(y_prob)).sum() / y_prob.shape[0]


class SGDOptimizer():

    def __init__(self, params, learning_rate_init=0.1,
                 momentum=0.9, nesterov=True):
        self.params = [param for param in params]
        self.learning_rate = float(learning_rate_init)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = [np.zeros_like(param) for param in params]

    def update_params(self, grads):
        updates = self._get_updates(grads)
        for param, update in zip(self.params, updates):
            param += update

    def _get_updates(self, grads):
        updates = [self.momentum * velocity - self.learning_rate * grad
                   for velocity, grad in zip(self.velocities, grads)]
        self.velocities = updates

        if self.nesterov:
            updates = [self.momentum * velocity - self.learning_rate * grad
                       for velocity, grad in zip(self.velocities, grads)]

        return updates


class AdamOptimizer():

    def __init__(self, params, learning_rate_init=0.001, beta_1=0.9,
                 beta_2=0.999, epsilon=1e-8):

        self.params = [param for param in params]
        self.learning_rate_init = float(learning_rate_init)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]

    def update_params(self, grads):
        updates = self._get_updates(grads)
        for param, update in zip(self.params, updates):
            param += update

        return self.params

    def _get_updates(self, grads):
        self.t += 1
        self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad
                   for m, grad in zip(self.ms, grads)]
        self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
                   for v, grad in zip(self.vs, grads)]
        self.learning_rate = (self.learning_rate_init *
                              np.sqrt(1 - self.beta_2 ** self.t) /
                              (1 - self.beta_1 ** self.t))
        updates = [-self.learning_rate * m / (np.sqrt(v) + self.epsilon)
                   for m, v in zip(self.ms, self.vs)]
        return updates




def gen_batches(n, batch_size, min_batch_size=0):
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        if end + min_batch_size > n:
            continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)



def accuracy_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape

    score = y_true == y_pred

    return np.average(score)


def label_binarize(y, classes):

    n_samples = y.shape[0]
    n_classes = len(classes)
    classes = np.asarray(classes)
    sorted_class = np.sort(classes)

    # binarizer label
    Y = np.zeros((n_samples, n_classes))
    indices = np.searchsorted(sorted_class, y)
    Y[np.arange(n_samples), indices] = 1

    return Y

