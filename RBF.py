# -*- coding: utf-8 -*-
"""

@author: Samane
"""
import math
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from numpy.linalg import norm, lstsq
from numpy import exp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
f_data = pd.read_csv("preswissroll.dat.txt",sep="   ", header=None, engine='python')
f_lable=pd.read_csv("preswissroll_labels.dat.txt", header=None)
total_data=np.hstack((f_data,f_lable))
np.random.shuffle(total_data)
print(total_data.shape)

# plt.figure(figsize=(10,8))
# ax=sns.scatterplot(x=total_data[:,0],
#                 y=total_data[:,1],
#                 hue=total_data[:,2],
#                 s=50,
#                 data=total_data)

# plt.xlabel("X", size=10)
# plt.ylabel("Y", size=10)


val_validation = float(input("Enter percent of validation set: ") )
X_train, X_test, y_train, y_test = train_test_split(total_data[:,0:2], total_data[:,2], test_size=0.0625,shuffle=True, random_state=4)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_validation,shuffle=True, random_state=4)

print("=====================Determine centers of the neurons using KMeans==========================")

k_list=[2,4,6,8,10,15,20,50,100]
strain=0
sval=0
stest=0

def function_name(X,k):
    if X == 1: 
        shape= X_train.shape
        row= shape[0]
        column= k
        G= np.empty((row,column), dtype= float)
        for i in range(row):
            for j in range(column):     
                dist= np.linalg.norm(X_train[i]-centers[j])
                G[i][j]=math.exp(-math.pow(dist,2)/math.pow(2*sigma,2))
                
      
                      
        GTG= np.dot(G.T,G)
        GTG_inv= np.linalg.inv(GTG)
        fac= np.dot(GTG_inv,G.T)
        W= np.dot(fac,y_train)
 
    
        row= X_test.shape[0]
        column= k
        G_test= np.empty((row,column), dtype= float)
        for i in range(row):
            for j in range(column):
                 dist=np.linalg.norm(X_test[i]-centers[j])
                 G_test[i][j]= math.exp(-math.pow(dist,2)/math.pow(2*sigma,2))


        row= X_val.shape[0]
        column= k
        G_val= np.empty((row,column), dtype= float)
        for i in range(row):
            for j in range(column):
                dist=np.linalg.norm(X_val[i]-centers[j])
                G_val[i][j]= math.exp(-math.pow(dist,2)/math.pow(2*sigma,2))
        print("****accuracy of Guassian with k=",k,"****")         
        prediction= np.dot(G,W)
        score_ytrain=accuracy_score(np.around(prediction),y_train)
        print("accuracy score of prediction and y_train: ",score_ytrain) 
        prediction= np.dot(G_test,W)
        score_ytest=accuracy_score(np.around(prediction),y_test)
        print("accuracy score of prediction and y_test: ",score_ytest) 
        prediction= np.dot(G_val,W)
        score_yval=accuracy_score(np.around(prediction),y_val)
        print("accuracy score of prediction and y_val: ",score_yval)
        
    if X == 2 : 
        shape= X_train.shape
        row= shape[0]
        column= k
        G= np.empty((row,column), dtype= float)
        for i in range(row):
            for j in range(column):     
                dist= np.linalg.norm(X_train[i]-centers[j])
                G[i][j]=math.sqrt(math.pow(sigma,2)+math.pow(dist,2))
      
                      
        GTG= np.dot(G.T,G)
        GTG_inv= np.linalg.inv(GTG)
        fac= np.dot(GTG_inv,G.T)
        W= np.dot(fac,y_train)
 
    
        row= X_test.shape[0]
        column= k
        G_test= np.empty((row,column), dtype= float)
        for i in range(row):
            for j in range(column):
                 dist=np.linalg.norm(X_test[i]-centers[j])
                 G_test[i][j]= math.sqrt(math.pow(sigma,2)+math.pow(dist,2))


        row= X_val.shape[0]
        column= k
        G_val= np.empty((row,column), dtype= float)
        for i in range(row):
            for j in range(column):
                dist=np.linalg.norm(X_val[i]-centers[j])
                G_val[i][j]= math.sqrt(math.pow(sigma,2)+math.pow(dist,2))
        print("****accuracy of Multiquadratic with k=",k,"****")         
        prediction= np.dot(G,W)
        score_ytrain=accuracy_score(np.around(prediction),y_train)
        print("accuracy score of prediction and y_train: ",score_ytrain) 
        prediction= np.dot(G_test,W)
        score_ytest=accuracy_score(np.around(prediction),y_test)
        print("accuracy score of prediction and y_test: ",score_ytest) 
        prediction= np.dot(G_val,W)
        score_yval=accuracy_score(np.around(prediction),y_val)
        print("accuracy score of prediction and y_val: ",score_yval)
        
        
    if X == 3: 
        shape= X_train.shape
        row= shape[0]
        column= k
        G= np.empty((row,column), dtype= float)
        for i in range(row):
            for j in range(column):     
                dist= np.linalg.norm(X_train[i]-centers[j])
                G[i][j]=math.pow((math.pow(sigma,2)+math.pow(dist,2)),-0.5)
      
                      
        GTG= np.dot(G.T,G)
        GTG_inv= np.linalg.inv(GTG)
        fac= np.dot(GTG_inv,G.T)
        W= np.dot(fac,y_train)
 
    
        row= X_test.shape[0]
        column= k
        G_test= np.empty((row,column), dtype= float)
        for i in range(row):
            for j in range(column):
                 dist=np.linalg.norm(X_test[i]-centers[j])
                 G_test[i][j]=math.pow((math.pow(sigma,2)+math.pow(dist,2)),-0.5)


        row= X_val.shape[0]
        column= k
        G_val= np.empty((row,column), dtype= float)
        for i in range(row):
            for j in range(column):
                dist=np.linalg.norm(X_val[i]-centers[j])
                G_val[i][j]= math.pow((math.pow(sigma,2)+math.pow(dist,2)),-0.5)
        print("****accuracy of InverseMultiquadratic with k=",k,"****")        
        prediction= np.dot(G,W)
        score_ytrain=accuracy_score(np.around(prediction),y_train)
        print("accuracy score of prediction and y_train: ",score_ytrain) 
        prediction= np.dot(G_test,W)
        score_ytest=accuracy_score(np.around(prediction),y_test)
        print("accuracy score of prediction and y_test: ",score_ytest) 
        prediction= np.dot(G_val,W)
        score_yval=accuracy_score(np.around(prediction),y_val)
        print("accuracy score of prediction and y_val: ",score_yval)
    return score_ytrain,score_yval,score_ytest



gussian_train=[]
gussian_val=[]
gussian_test=[]
    
multiquad_train=[]
multiquad_val=[]
multiquad_test=[]
    
invmulti_train=[]
invmulti_val=[]
invmulti_test=[]    
for k in k_list:
    clusterer=KMeans(n_clusters= k).fit(X_train)
    centers= clusterer.cluster_centers_
    max=0    
    for i in range(k):
        for j in range(k):
            if i != j:
                d= np.linalg.norm(centers[i]-centers[j])
                if(d> max):
                    max= d
    d= max                
    sigma= np.max(d) / np.sqrt(2*k)
    
    strain,sval,stest=function_name(1,k)
    gussian_train.append(strain)
    gussian_val.append(sval)
    gussian_test.append(stest)
    
    strain,sval,stest=function_name(2,k)
    multiquad_train.append(strain)
    multiquad_val.append(sval)
    multiquad_test.append(stest)
    
    strain,sval,stest=function_name(3,k)
    invmulti_train.append(strain)
    invmulti_val.append(sval)
    invmulti_test.append(stest)  
    
# Gaussian

gussian_train=np.asarray(gussian_train).reshape((9,1))
gussian_val=np.asarray(gussian_val).reshape((9,1))
gussian_test=np.asarray(gussian_test).reshape((9,1))

plt.figure()
plt.title("Gaussian RBF")
plt.plot(k_list,gussian_train,'c',label="Train")
plt.plot(k_list,gussian_val,'y',label="Valication")
plt.plot(k_list,gussian_test,'red',label="Test")

plt.xlabel('Number of K')
plt.ylabel("Score")
plt.legend(loc='best')

# Multi

multiquad_train=np.asarray(multiquad_train).reshape((9,1))
multiquad_val=np.asarray(multiquad_val).reshape((9,1))
multiquad_test=np.asarray(multiquad_test).reshape((9,1))

plt.figure()
plt.title("Multi Quadratic RBF")
plt.plot(k_list,multiquad_train,'c',label="Train")
plt.plot(k_list,multiquad_val,'y',label="Valication")
plt.plot(k_list,multiquad_test,'red',label="Test")

plt.xlabel('Number of K')
plt.ylabel("Score")
plt.legend(loc='best')

# Inverse Multi

invmulti_train=np.asarray(invmulti_train).reshape((9,1))
invmulti_val=np.asarray(invmulti_val).reshape((9,1))
invmulti_test=np.asarray(invmulti_test).reshape((9,1))

plt.figure()
plt.title("Inverse Multi Quadratic RBF")
plt.plot(k_list,invmulti_train,'c',label="Train")
plt.plot(k_list,invmulti_val,'y',label="Valication")
plt.plot(k_list,invmulti_test,'red',label="Test")

plt.xlabel('Number of K')
plt.ylabel("Score")
plt.legend(loc='best')







# Gaussian
rbf_list=["Gaussian","Multi Quadratic","Inv_Multi Quadratic"]
train_list=[]
val_list=[]
test_list=[]
strain,sval,stest=function_name(1,50)
train_list.append(strain)
val_list.append(sval)
test_list.append(stest)

strain,sval,stest=function_name(2,50)
train_list.append(strain)
val_list.append(sval)
test_list.append(stest)

strain,sval,stest=function_name(3,50)
train_list.append(strain)
val_list.append(sval)
test_list.append(stest)


train_list=np.asarray(train_list).reshape((3,1))
val_list=np.asarray(val_list).reshape((3,1))
test_list=np.asarray(test_list).reshape((3,1))

plt.figure()
plt.title("Functions Compare")
plt.plot(rbf_list,train_list,'C3',label="Train")
plt.plot(rbf_list,val_list,'C4',label="Valication")
plt.plot(rbf_list,test_list,'C5',label="Test")
plt.xlabel('Functions')
plt.ylabel("Score")
plt.legend(loc='best')



plt.show()

    
