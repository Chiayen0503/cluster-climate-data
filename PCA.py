#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:47:10 2018

@author: ChiaYen
"""

import numpy as np

import scipy as sc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

dataRaw = [];
DataFile = open("ClimateData.csv","r")
while True:
    theline = DataFile.readline()
    if len(theline)== 0:
        break
    readData = theline.split(",")
    for pos in range(len(readData)):
        readData[pos] = float(readData[pos]);
    dataRaw.append(readData)

DataFile.close()
data = np.array(dataRaw)

def standard(data):
    standardData = data.copy()

    rows = data.shape[0]
    cols = data.shape[1]

    for j in range(cols):
        sigma = np.std(data[: , j ])
        mu = np.mean( data[: , j ])

        for i in range(rows):
            standardData[i , j] = (data [i , j]-mu) / sigma
    return standardData



def plot(data, fileName ):
    fig,((ax1 , ax2 ),( ax3 , ax4 )) = plt.subplots(nrows =2 , ncols =2)

    fig.set_size_inches (9.0 ,6.0)

    ax1.plot( data [: ,0] , data [: ,1] ,".")
    ax2.plot( data [: ,0] , data [: ,2] ,".")
    ax3.plot( data [: ,0] , data [: ,3] ,".")
    ax4.plot( data [: ,0] , data [: ,4] ,".")

    ax1.set_ylabel(" Wind Speed ")
    ax2.set_ylabel (" Wind Direction ")
    ax3.set_ylabel (" Precipitation ")
    ax4.set_ylabel (" Humidity ")

    ax1.set_xlabel (" Temperature ")
    ax2.set_xlabel (" Temperature ")
    ax3.set_xlabel (" Temperature ")
    ax4.set_xlabel (" Temperature ")
    
    plt.savefig(fileName,bbox_inches ='tight')
    
plot (data, "notStandardised.pdf" )
plot (standard(data), "standardised.pdf")


#PCA
def centralize(data):
    centralizedData = data.copy()
    rows = data.shape[0]
    cols = data.shape[1]
    for j in range(cols):
        mu = np.mean(data[:,j])
        for i in range(rows):
            centralizedData[i,j] = (data[i,j] - mu)
    return centralizedData

centralizedData = centralize(data)
    
np.corrcoef(centralizedData ,rowvar=False)


pca = PCA(n_components=5)
pca.fit(centralizedData)
Coeff = pca.components_

transformedData = pca.transform(centralizedData)
plt.figure(figsize=(6,4))
plt.plot(transformedData[:,0],transformedData[:,1],".")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.savefig("PCAData.pdf")


plt.figure(figsize=(6,4))
plt.bar([1,2,3,4,5],
        pca.explained_variance_ratio_ ,tick_label=[1,2,3,4,5])
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")
plt.savefig("PCAAnalysis.pdf")



## Exercise 5 solution: order according to eigenValues
cov = np.cov(centralizedData,rowvar=False)

eigVals, eigVectors = sc.linalg.eig(cov)


orderedEigVectors = np.empty(eigVectors.shape)

tmp = eigVals.copy()

maxValue = float("-inf")
maxValuePos = -1

for i in range(len(eigVectors)):

    maxValue = float("-inf")
    maxValuePos = -1
        
    for n in range(len(eigVectors)):
        if (tmp[n] > maxValue):
            maxValue = tmp[n]
            maxValuePos = n

    orderedEigVectors[:,i] = eigVectors[:,maxValuePos]
    tmp[maxValuePos] = float("-inf")

k = 2

# orderedEigVectors[:,1] = -orderedEigVectors[:,1] ## This inversion will generate the same graph as the one using the library

projectionMatrix = orderedEigVectors[:,0:k]

pcaByHandData = centralizedData.dot(projectionMatrix)

plt.figure(figsize=(6,4))

plt.plot(pcaByHandData[:,0],pcaByHandData[:,1],".")

plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")

plt.savefig("PCAByHandData.pdf")

plt.close()
