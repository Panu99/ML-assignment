#!/usr/bin/env python
# coding: utf-8

# ML_Assignment(K-Means Algorithm)
# Name:PranavHegde
# USN:1MS18IS412

# In[21]:


import os
import struct
import numpy 
import pandas 
import sklearn.datasets
import ipyvolume as ipv


# In[4]:


from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from ipywidgets import ColorPicker, VBox,     interact, interactive, fixed


# In[6]:


def compute_bic(kmeans,x):

    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    m = kmeans.n_clusters
    n = numpy.bincount(labels)
    N, d = x.shape
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(x[numpy.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])
    const_term = 0.5 * m * numpy.log(N) * (d+1)
    BIC = numpy.sum([n[i] * np.log(n[i]) -
               n[i] * numpy.log(N) -
             ((n[i] * d) / 2) * numpy.log(2*numpy.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)


# In[7]:


url = 'https://raw.githubusercontent.com/MSPawanRanjith/FileTransfer/master/kmean_dataset.csv'
df = pandas.read_csv(url, error_bad_lines=False)
df.head(12)


# In[9]:


x = df.iloc[:, [0,1,2]].values
kmeans5 = KMeans(n_clusters=5)
y_kmeans5 = kmeans5.fit_predict(x)
print(y_kmeans5)


# In[10]:


kmeans5.cluster_centers_


# In[12]:


Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()


# In[13]:


print("the changes are in 2 and 4 are: & we are taking average value")


# In[14]:


kmeans3 = KMeans(n_clusters=3)
y_kmeans3 = kmeans3.fit_predict(x)
print(y_kmeans3)

kmeans3.cluster_centers_


# In[15]:


plt.scatter(x[:,0],x[:,1], c=y_kmeans3,cmap='rainbow')


# In[16]:


print("To find the value of BIC using BIC Function")


# In[17]:


ks = range(1,9)
KMeans = [cluster.KMeans(n_clusters = i, init="k-means++").fit(x) for i in ks]
BIC = [compute_bic(kmeansi,x) for kmeansi in KMeans]
print(BIC)


# In[18]:


print (max(BIC))


# In[20]:


print("the optimum value from Using BIC",BIC.index(max(BIC))+1)


# In[31]:


plt.plot(ks,BIC,'r-o')
plt.title(" data  (k vs BIC)")
plt.xlabel("# k values")
plt.ylabel("# BIC")


# In[ ]:




