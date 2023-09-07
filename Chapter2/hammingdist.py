# Function for Hamming distance calculation

def hamming_distance(s1, s2):
  dist = 0
  if len(s1)!=len(s2):
      print("Strings are not equal")
  else:
      for x, (i, j) in enumerate(zip(s1, s2)):
          if i != j:
              print(f'Characters do not match {i,j} in {x}')
              dist+=1
  return f'Hamming Distance = {dist}'

A = "Euclidean"
B = "Manhattan"

hamming = hamming_distance(A, B)
print(hamming)