# importing numpy library
import numpy as np

# creating row vectors
a = np.array([2, 4, 6])
print("Row vector a: ",a)
b = np.array([1, 2, 3])
print("Row vector b: ",b)

# creating column vectors
s = np.array([[2], 
             [4], 
             [6]])
print("Column vector s: ",s)
t = np.array([[1], 
             [2], 
             [3]])
print("Column vector t: ",t)

# addition
c = a + b
print("Addition: ",c)

# substraction
d = a - b
print("Substraction: ",d)

# mutiplication
e = a * b
print("Multiplication: ",e)

# division
f = a / b
print("Division: ",f)

# dot product
g = a.dot(b)
print("Dot product: ",g)

# scalar multiplication
h = 0.5 * a
print("Scalar multiplication", h)