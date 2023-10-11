
#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing Sci-Kit Learn Algorithms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Data collection
har=pd.read_csv("/home/gopi/Desktop/Mini_Project/train.csv")
har_test =pd.read_csv("/home/gopi/Desktop/Mini_Project/test.csv")

#print(har_train.head())
print("---------------------------------------------------------")
#print("Number of Rows and column of the data set:",har_train.shape)
#print("Number of Columns :",har_train.columns)
print("---------------------------------------------------------")
#print("Information of training dataset ",har_train.info())
print("_______________________________________________________________")
#print("Information of testing dataset ",har_test.info())

#concadinating two data set
#har=pd.concat([har_train,har_test],ignore_index=True)
#print(har.shape)

#preprossing
print(har.duplicated().any())
print(har.isnull().sum())
"""
#plotting Bar Graph
sns.countplot(x="Activity",data=har)
plt.xticks(rotation=15)
plt.show()
"""
#encoding
encode=LabelEncoder()
har["Activity"]=encode.fit_transform(har["Activity"])
print(har["Activity"])

#Splitting dataset 
x=har.drop("Activity",axis=1)
y=har["Activity"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=80)

#Scaling the given data
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
"""
#logistic Regression
logistic=LogisticRegression(C=0.03, solver="saga", max_iter=100)
logistic.fit(x_train,y_train)
log_pred=logistic.predict(x_test)
print(f"Accuracy of Logistic Regression :  {accuracy_score(y_test,log_pred)}")
print("-------------------------------------------------------")
"""
#Random Forest
random=RandomForestClassifier(n_estimators=100,random_state=42)
random.fit(x_train,y_train)
rf_pred=random.predict(x_test)
print(f"Accuracy of RandomForest : {accuracy_score(y_test,rf_pred)}")
print("---------------------------------------------------------")
"""
#KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
knn_pred=knn.predict(x_test)
print(f"Accuracy of KNN : {accuracy_score(y_test,knn_pred)}")
print("--------------------------------------------------------")
"""

"""
#Confusion Matrix
matrix1=confusion_matrix(y_test,log_pred)
cm_display1=ConfusionMatrixDisplay(confusion_matrix=matrix1, display_labels=[])
cm_display1.plot()
plt.title("\nConfusion Matrix :Logistic Regression\n")
plt.show()

matrix2=confusion_matrix(y_test,rf_pred)
cm_display2=ConfusionMatrixDisplay(confusion_matrix=matrix2, display_labels=[])
cm_display2.plot()
plt.title("\nConfusion Matrix : RandomForest\n")
plt.show()

matrix=confusion_matrix(y_test,knn_pred)
cm_display=ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[])
cm_display.plot()
plt.title("\nConfusion Matrix : KNN\n")
plt.show()
"""

#decode after prediction
print(rf_pred)
print(type(rf_pred))
rf_pred=encode.inverse_transform(rf_pred)
print(rf_pred)

#predicted values convert to csv file
pred_df=pd.DataFrame(data={"Predicted_activity":rf_pred})
pred_df.to_csv("Har.csv",index=False)

