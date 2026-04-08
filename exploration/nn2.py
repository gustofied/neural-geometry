import numpy as np

logits = np.array([2.0, 1.0, 0.1])
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

probabilties = softmax(logits)

print(f"Softax probabilites: {probabilties}")
print(f"Sum: {np.sum(probabilties)}")


true_label = np.array([1, 0 ,0])

def cross_entropy(predictions, labels):
    return -np.sum(labels * np.log(predictions + 1e-15))
    # The sum selects the correct class since labels are one-hot, meh implemtnation
    # leaving just -log(p_correct)

loss = cross_entropy(probabilties, true_label)

print(f"Loss: {loss}")
