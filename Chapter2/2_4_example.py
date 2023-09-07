import numpy as np
from numpy.linalg import norm

# Euclidean Distance function
def euclidean_distance(A, B):

  A, B = np.array(A), np.array(B)
  sum = np.sum(np.square(A - B))

  return np.sqrt(sum)

# Cosine Similarity function
def cosine_similarity(A, B):

  A, B = np.array(A), np.array(B)
  cosine = np.dot(A, B)/(norm(A)*norm(B))

  return cosine

# Implement the rest of metrics yourself
  
A = [1, 3, 5, 7, 9]
B = [2, 4, 6, 8, 12]

euc = euclidean_distance(A=A, B=B)
cos = cosine_similarity(A=A, B=B)
d_cos = 1 - cos                         # Cosine distance = 1-cos(theta)

print("Euclidean Distance = {}\nCosine Similarity = {}\nCosine distance = {}".format(euc, cos, d_cos))