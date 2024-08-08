from neuralnet import Perceptron
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from utils import calculate_accuracy
import matplotlib.pyplot as plt
import random

digit_data = load_digits()
X, y = digit_data.data, digit_data.target

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

perceptron = Perceptron(32, 10)
epochs, losses, accuracies = perceptron.fit(X_train, y_train, alpha=0.1, epochs=500)
y_predictions = perceptron.predict(X_test)

accuracy = calculate_accuracy(y_predictions, y_test)

print("Accuracy:", accuracy)

# plt.title("Alpha Learning Rate = 0.1")
# plt.xlabel("Epochs")
# plt.plot(epochs, losses, label="Loss")
# plt.plot(epochs, accuracies, label="Accuracy")
# plt.axhline(y=1, color='black', linestyle=':')
# plt.legend()
# plt.show()

r_count = 3
c_count = 5
fig, ax = plt.subplots(nrows=r_count, ncols=c_count, figsize=(12, 8))
for r in range(r_count):
    for c in range(c_count):
        pos = random.randint(0, y_test.size)
        ax[r, c].imshow(X_test[pos, :].reshape((8, 8)))
        ax[r, c].set_title(f"Predicted: {y_predictions[pos]}; Actual: {y_test[pos]}", fontsize=10)
        ax[r, c].axis('off')
plt.show()