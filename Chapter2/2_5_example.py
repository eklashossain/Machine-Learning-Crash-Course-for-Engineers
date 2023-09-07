# Importing Natural Language Toolkit library
# NLTK associated packages are downloaded

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def jaccard_similarity(s1, s2):

  # Tokenizing sentences, i. e., splitting the sentences into words
  S1_list = word_tokenize(s1)
  S2_list = word_tokenize(s2)

  # Getting the English stopword collection
  sw = stopwords.words('english')

  # Creating word sets corresponding to each sentence
  S1_set = {word for word in S1_list if not word in sw}
  S2_set = {word for word in S2_list if not word in sw}

  print(f'Word set Sentence 1 = {S1_set}')
  print(f'Word set Sentence 2 = {S2_set}')

  I = set(S1_set).intersection(set(S2_set))     # Intersection operation
  U = set(S1_set).union(set(S2_set))            # Union operation

  print(f'Intersection = {I}')
  print(f'Union = {U}')

  IoU = len(I)/len(U)                           # Intersection over Union (IoU)

  return IoU

A = "This is a foo bar sentence ."
B = "This sentence is similar to a foo bar sentence ."

J = jaccard_similarity(A, B)
d_J = 1 - J                                             # Jaccard distance = 1 - J
print("Jaccard Similarity Index = {}\nJaccard distance = {}".format(J, d_J))