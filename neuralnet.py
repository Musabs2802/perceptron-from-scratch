import numpy as np


class Perceptron:
    def __init__(self, l1, l2) -> None:
        self.l1 = l1
        self.l2 = l2
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
    
    def fit(self, X, Y, alpha=0.1, epochs=100):
        self.W1 = np.random.randn(self.l1, X.shape[1]) * 0.01
        self.b1 = np.zeros((self.l1, 1))

        self.W2 = np.random.randn(self.l2, self.l1) * 0.01
        self.b2 = np.zeros((self.l2, 1))

        self.alpha = alpha
        self.epochs = epochs

        return self.__train(X, Y)
        
    def predict(self, X):
        _, _, _, y = self.__forward_propagation(X)
        return self.__get_predictions(y)
    
    def __train(self, X, Y):
        epochs_list, losses, accuracies = [], [], []

        for i in range(self.epochs):
            Z1, A1, Z2, A2 = self.__forward_propagation(X)
            dW1, db1, dW2, db2 = self.__back_propagation(X, Y, Z1, A1, Z2, A2)
            self.__update_params(dW1, db1, dW2, db2)

            if i % 5 == 0:
                loss = self.__calculate_loss(A2, Y)
                accuracy = self.__get_accuracy(self.__get_predictions(A2), Y)
                epochs_list.append(i)
                losses.append(loss)
                accuracies.append(accuracy)
                print("Epoch:", i, "Loss:", loss, "Accuracy:", accuracy)

        return epochs_list, losses, accuracies
    
    def __forward_propagation(self, X):
        Z1 = np.dot(self.W1, X.T) + self.b1
        A1 = self.__RELU(Z1)

        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.__softmax(Z2)

        return Z1, A1, Z2, A2

    def __back_propagation(self, X, Y, Z1, A1, Z2, A2):
        m = X.size

        dZ2 = A2 - self.__one_hot(Y)
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = (1/m) * np.dot(dZ1, X)
        db1 = (1/m) * np.sum(dZ1)

        return dW1, db1, dW2, db2

    def __update_params(self, dW1, db1, dW2, db2):
        self.W1 = self.W1 - self.alpha * dW1
        self.b1 = self.b1 - self.alpha * db1
        self.W2 = self.W2 - self.alpha * dW2
        self.b2 = self.b2 - self.alpha * db2
    
    def __calculate_loss(self, A, y):
        m = y.shape[0]
        log_probs = -np.log(A[y, range(m)])
        loss = np.sum(log_probs) / m
        return loss
    
    def __get_predictions(self, A):
        return np.argmax(A, 0)
    
    def __get_accuracy(self, y_predictions, y_actuals):
        return np.sum(y_predictions == y_actuals) / len(y_actuals)
    
    def __one_hot(self, y):
        one_hot = np.zeros((y.size, y.max() + 1))
        one_hot[np.arange(y.size), y] = 1
        return one_hot.T

    def __softmax(self, Z):
        return np.exp(Z) / sum(np.exp(Z))
    
    def __RELU(self, Z):
        return np.maximum(0, Z)