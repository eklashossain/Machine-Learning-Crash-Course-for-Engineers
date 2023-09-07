import numpy as np

def categorical_cross_entropy(y, y_hat):
  
  y, y_hat = np.array(y), np.array(y_hat)
  ce = -np.sum(np.multiply(y,np.log10(y_hat)))

  return ce

y = [0, 1, 0]
y_hat = [0.02,0.97835,0.00165]
ce_loss = categorical_cross_entropy(y=y, y_hat=y_hat)

print("Cross-Entropy Loss = {}".format(ce_loss))