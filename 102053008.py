import pandas as pd
import numpy as np
import random
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data=pd.read_csv('D:\\uday\\uday\\semester 6\\Predictive Analysis using Statistics\\Assignments\\Assignment 3\\Sampling_Assignment\\Creditcard_data.csv')
# print('The Class Distribution of the data is : ',data.loc[:,'Class'].value_counts())

#Oversampling
# oversample = RandomOverSampler(sampling_strategy='minority')
oversample = SMOTE(sampling_strategy='minority',random_state=42)
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x_sampled,y_sampled=oversample.fit_resample(x,y)
# print(y_sampled.value_counts())
data2=pd.concat([x_sampled,y_sampled], axis=1)
# print(data2)

#Simple Random Sampling
n_sample_simpleRandomSampling=(1.96**2)*0.5*(1-0.5)/(0.05**2)
data_simpleRandomSampling=(data2.sample(int(n_sample_simpleRandomSampling),random_state=42))
# print(data_simpleRandomSampling)

x_train, x_test, y_train, y_test=train_test_split(data_simpleRandomSampling.iloc[:,:-1],data_simpleRandomSampling.iloc[:,-1],test_size=0.2,random_state=42)
m1 = LogisticRegression(max_iter= 2500,random_state=42)
m2 = DecisionTreeClassifier(random_state=42)
m3 = SVC(random_state=42)
m4 = KNeighborsClassifier()
m5 = GaussianNB()

m1.fit(x_train,y_train)
m2.fit(x_train,y_train)
m3.fit(x_train,y_train)
m4.fit(x_train,y_train)
m5.fit(x_train,y_train)

y_pred1 = m1.predict(x_test)
y_pred2 = m2.predict(x_test)
y_pred3 = m3.predict(x_test)
y_pred4 = m4.predict(x_test)
y_pred5 = m5.predict(x_test)

result=[[accuracy_score(y_test, y_pred1),accuracy_score(y_test, y_pred2),accuracy_score(y_test,y_pred3),accuracy_score(y_test,y_pred4),accuracy_score(y_test,y_pred5)]]

#Systematic Random Sampling
data_systematic=data2.iloc[[i for i in range(5,1000,2)],:]

x_train, x_test, y_train, y_test=train_test_split(data_systematic.iloc[:,:-1],data_systematic.iloc[:,-1],test_size=0.2,random_state=42)
m1 = LogisticRegression(max_iter= 2500,random_state=42)
m2 = DecisionTreeClassifier(random_state=42)
m3 = SVC(random_state=42)
m4 = KNeighborsClassifier()
m5 = GaussianNB()

m1.fit(x_train,y_train)
m2.fit(x_train,y_train)
m3.fit(x_train,y_train)
m4.fit(x_train,y_train)
m5.fit(x_train,y_train)

y_pred1 = m1.predict(x_test)
y_pred2 = m2.predict(x_test)
y_pred3 = m3.predict(x_test)
y_pred4 = m4.predict(x_test)
y_pred5 = m5.predict(x_test)

res=[accuracy_score(y_test, y_pred1),accuracy_score(y_test, y_pred2),accuracy_score(y_test,y_pred3),accuracy_score(y_test,y_pred4),accuracy_score(y_test,y_pred5)]
result.append(res)

#stratified sampling
# print(data2)
n_sample_StratifiedSampling=(1.96**2)*0.3*(1-0.3)/((0.05/2)**2)
# print(n_sample_StratifiedSampling)
data_stratified=data2.groupby('Class', group_keys=False).apply(lambda x: x.sample(int(n_sample_StratifiedSampling/2),random_state=42))
# print(data_stratified)
x_train, x_test, y_train, y_test=train_test_split(data_stratified.iloc[:,:-1],data_stratified.iloc[:,-1],test_size=0.2,random_state=42)
m1 = LogisticRegression(max_iter= 2500,random_state=42)
m2 = DecisionTreeClassifier(random_state=42)
m3 = SVC(random_state=42)
m4 = KNeighborsClassifier()
m5 = GaussianNB()

m1.fit(x_train,y_train)
m2.fit(x_train,y_train)
m3.fit(x_train,y_train)
m4.fit(x_train,y_train)
m5.fit(x_train,y_train)

y_pred1 = m1.predict(x_test)
y_pred2 = m2.predict(x_test)
y_pred3 = m3.predict(x_test)
y_pred4 = m4.predict(x_test)
y_pred5 = m5.predict(x_test)

res=[accuracy_score(y_test, y_pred1),accuracy_score(y_test, y_pred2),accuracy_score(y_test,y_pred3),accuracy_score(y_test,y_pred4),accuracy_score(y_test,y_pred5)]
result.append(res)

#cluster sampling
n_sample_ClusterSampling=(1.96**2)*0.1*(1-0.1)/((0.05/3)**2)
s=set(list(data2['Time']))
s1=pd.Series(list(s))
data_clustered=(data2[data2['Time'].isin([ i for i in s1.sample(int(n_sample_ClusterSampling/3),random_state=42)])])

x_train, x_test, y_train, y_test=train_test_split(data_clustered.iloc[:,:-1],data_clustered.iloc[:,-1],test_size=0.2,random_state=42)
m1 = LogisticRegression(max_iter= 2500,random_state=42)
m2 = DecisionTreeClassifier(random_state=42)
m3 = SVC(random_state=42)
m4 = KNeighborsClassifier()
m5 = GaussianNB()

m1.fit(x_train,y_train)
m2.fit(x_train,y_train)
m3.fit(x_train,y_train)
m4.fit(x_train,y_train)
m5.fit(x_train,y_train)

y_pred1 = m1.predict(x_test)
y_pred2 = m2.predict(x_test)
y_pred3 = m3.predict(x_test)
y_pred4 = m4.predict(x_test)
y_pred5 = m5.predict(x_test)

res=[accuracy_score(y_test, y_pred1),accuracy_score(y_test, y_pred2),accuracy_score(y_test,y_pred3),accuracy_score(y_test,y_pred4),accuracy_score(y_test,y_pred5)]
result.append(res)


#Quota Sampling
data_only0=data2[data2['Class']==0].iloc[:500]
data_only1=data2[data2['Class']==1].iloc[:500]
data_quotasampling =pd.concat([data_only0 ,data_only1], axis=0)

x_train, x_test, y_train, y_test=train_test_split(data_quotasampling.iloc[:,:-1],data_quotasampling.iloc[:,-1],test_size=0.2,random_state=42)
m1 = LogisticRegression(max_iter= 2500,random_state=42)
m2 = DecisionTreeClassifier(random_state=42)
m3 = SVC(random_state=42)
m4 = KNeighborsClassifier()
m5 = GaussianNB()

m1.fit(x_train,y_train)
m2.fit(x_train,y_train)
m3.fit(x_train,y_train)
m4.fit(x_train,y_train)
m5.fit(x_train,y_train)

y_pred1 = m1.predict(x_test)
y_pred2 = m2.predict(x_test)
y_pred3 = m3.predict(x_test)
y_pred4 = m4.predict(x_test)
y_pred5 = m5.predict(x_test)

res=[accuracy_score(y_test, y_pred1),accuracy_score(y_test, y_pred2),accuracy_score(y_test,y_pred3),accuracy_score(y_test,y_pred4),accuracy_score(y_test,y_pred5)]
result.append(res)

print(result)
max1=-1
st=""
ml=""
for i in range(len(result)):
    for j in range(len(result[0])):
        if (result[i][j]>max1):          
            if i==0:
                st="Simple Random Sampling"
            elif i==1:
                st="Systematic Random Sampling"
            elif i==2:
                st="Stratified sampling"
            elif i==3:
                st="Cluster Sampling"
            elif i==4:
                st="Quota Sampling"
            if j==0:
                ml="Logistic Regression"
            elif j==1:
                ml="Decision Tree Classifier"
            elif j==2:
                ml="Support Vector Classifier"
            elif j==3:
                ml="KNeighbors Classifier"
            elif j==4:
                ml="Gaussian Naive Bayes"                    
            max1=result[i][j]

print(f'{ml} method with {st} gives the highest accuracy {max1}') 


