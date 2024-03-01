import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

df=pd.read_csv(r"C:\Users\STUDENT\Downloads\Social_Network_Ads.csv")
x = df.iloc[:,2:4]
y = df.iloc[:,4:5]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2) 
LogisticR=LogisticRegression()
LogisticR.fit(x_train,y_train)
predicition=LogisticR.predict(x_test)
plt.plot(x_test)
