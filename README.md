# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.
2.Analyse the data.
3.Use modelselection and Countvectorizer to preditct the values.
4.Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Hemasonica.P
RegisterNumber:  212222230048
*/

import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd 
data = pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()



x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)


from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)


y_pred = svc.predict(x_test)
y_pred


from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

# Result
![image](https://github.com/Hemasonica774/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118361409/9e942fa2-866c-4138-86e8-5c58d60d98e6)

#data.head()
![image](https://github.com/Hemasonica774/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118361409/fe006057-932e-46b8-88c3-bded7271e4b4)

#data.info()
![image](https://github.com/Hemasonica774/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118361409/8debf9f1-1bb6-4a87-bb7e-7e4ea51fce53)

#data.isnull().sum()
![image](https://github.com/Hemasonica774/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118361409/f6244309-d312-47d0-8b34-9edcacf1f1cb)

#Y_prediction value
![image](https://github.com/Hemasonica774/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118361409/27f032aa-788e-4b84-b3d6-8e9c33db147e)


# Accuracy value
![image](https://github.com/Hemasonica774/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118361409/31dc8b61-078f-49c2-9026-2388ce36b2bf)





## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
