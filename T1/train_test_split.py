import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
FIRST_CSV = 3
SECOND_CSV = 2

df = pd.read_csv('algebra.csv')
X = df.iloc[:,:2]
y = df.iloc[:,SECOND_CSV]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train = pd.concat([X_train,y_train],axis=1)

X_test = pd.concat([X_test,y_test],axis=1)
X_test.to_csv('test2.csv',header=False)
#X_train.to_csv('train2.csv',header = False)
#print(X_train)
