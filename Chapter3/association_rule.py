#Data: https://github.com/viktree/curly-octo-chainsaw/blob/master/BreadBasket_DMS.csv

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

#Loading dataset
dataset = pd.read_csv('data/BreadBasket_DMS.csv')

print(dataset.head(10))

#Dropping Duplicate Transaction
dataset = dataset.drop_duplicates()

#Taking Date, Time, Transaction and Item columns
print(dataset[['Date', 'Time', 'Transaction', 'Item']].head(10))

#Convert transacton & item into Crosstab
transaction = pd.crosstab(index= dataset['Transaction'], columns= dataset['Item'])
print(transaction.head(10))

#Removing "NONE"
transaction = transaction.drop(['NONE'], axis = 1)

print(transaction.head(10))

#Frequent itemset with min support = 4%
frequent_itemset = apriori(df = transaction, min_support= 0.04, use_colnames= True)
frequent_itemset.sort_values(by = 'support', ascending = False)

#Rule with minimun confidence = 50%
Rules = association_rules(frequent_itemset, min_threshold= 0.5)
print(Rules.head())

#Sorting results by lift to get highly associated itemsets
print(Rules.sort_values(by='lift', ascending= False).head())