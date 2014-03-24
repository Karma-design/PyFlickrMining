# -*- coding: utf-8 -*-
print(__doc__)

# Code source: Karim Alaoui
# License: BSD 3 clause

#HOW TO USE :
# 1.Set the input and output filenames
input_filename = 'lyon_cleaned_url_100' ##possible values : 'lyon_cleaned','suburb_cleaned','all_cleaned'
output_filename = 'clustering_v2'
#Set the pictures ratio : 1/(pictures ratio) pictures are displayed
pictRatio = 15

# 2.Comment the ununsed code (MeanShift or Kmeans, with plot)
# 3.If you use KMeans, don't forget to set the number of clusters
# NB : Clustering of large amount of data may take some time to perform using MeanShift

import pandas as pd
import numpy as np
import os

input_path = os.path.dirname(os.path.realpath(__file__))+'/%s.csv' 
path = input_path% input_filename

df = pd.read_csv(path)
Xa = df[['latitude', 'longitude_normalized', 'longitude', 'user','First*(id)','First*(url)']].values #longitude_normalized on 2pos when possible
latitudeIdx = 0
longitudeNrmIdx = 1
longitudeIdx = 2
userIdx = 3
idIdx = 4
urlIdx = 5

###############################################################################
# Compute clustering with MeanShift
from sklearn.cluster import MeanShift

# The following bandwidth can be automatically detected using
bandwidth = 0.0022

ms = MeanShift(bandwidth=bandwidth,bin_seeding=True, cluster_all=False, min_bin_freq=15)
X = Xa[:, 0:2]

ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)-1

print("number of estimated clusters : %d" % n_clusters_)

##############################
# Plot result
import pylab as pl
from itertools import cycle

pl.figure(1)
pl.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    pl.plot(X[my_members, 0], X[my_members, 1], col + '.')
    pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=14)
pl.title('Estimated number of clusters: %d' % n_clusters_)
pl.show()

####CLUSTERS JSON
json="var msg = '["
color = ['red','blue','purple','yellow','green','lightblue','orange','pink']
for k in range(n_clusters_):
    if k != 0:
        json += ","
    json += "{"
    ##Longitude
    json +="\"longitude\":"
    my_members = labels == k
    cluster_center = cluster_centers[k]
    json += str(round(cluster_center[1]/1.43,4))
    #json += cluster_center[1].astype('|S6')
    json += ", "
    ##Latitude
    json +="\"latitude\":"
    my_members = labels == k
    cluster_center = cluster_centers[k]
    json += cluster_center[0].astype('|S6')
    json += ", "
    ##Color
    json +="\"color\":\""
    json += color[k%8]
    json += "\""
    ##
    json += "}"
    
json += "]'; \n\n "

####

###PICTURES JSON

json+="var donnees = '["
for k in range(n_clusters_):
    my_members = labels == k
    for l in range(X[my_members,0].size/pictRatio):
        if l+k != 0:
            json += ","
        json += "{"
        ##Longitude
        json +="\"longitude\":"
        array = Xa[my_members, longitudeIdx]
        #json += str(cluster_center[1]/1.43)
        json += str(array[l])#.astype('|S6')
        json += ", "
        ##Latitude
        json +="\"latitude\":"
        array = Xa[my_members, latitudeIdx]
        json += str(array[l])#.astype('|S6')
        json += ", "
        ##Color
        json +="\"color\":\""
        json += color[k%8]
        json += "\""
        json += ", "
        ##Id
        json +="\"id\":"
        array = Xa[my_members, idIdx]
        json += str(array[l])#.astype('|S6')
        json += ", "
        ##url
        json +="\"url\":\""
        array = Xa[my_members, urlIdx]
        json += str(array[l])#.astype('|S6')
        json += "\", "
        ##User
        json +="\"user\":\""
        array = Xa[my_members, userIdx]
        json += array[l]
        json += "\"}"

json += "]';"

#Writing to text file
with open(os.path.dirname(os.path.realpath(__file__))+'/res/begin.html', 'r') as text_file:
    begin=text_file.read()
with open(os.path.dirname(os.path.realpath(__file__))+'/res/end.html', 'r') as text_file:
    end=text_file.read()

with open(os.path.dirname(os.path.realpath(__file__))+'/'+output_filename+'.html', "w") as text_file:
    #Static file start
    text_file.write("{0}".format(begin))
    #Writing generated content
    text_file.write("{0}".format(json))
    #Static file ending
    text_file.write("{0}".format(end))
    
###END OTHER JSON
###############################################################################

'''
###############################################################################
#Compute clustering with Kmeans
kmeans_n_clusters = 50

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=kmeans_n_clusters, n_init=10)
kmeans.fit(X)

##############################
# Plot Kmeans result
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

from matplotlib import pyplot
import numpy as np

for i in range(kmeans_n_clusters):
    # select only data observations with cluster label == i
    ds = X[np.where(labels==i)]
    # plot the data observations
    pyplot.plot(ds[:,0],ds[:,1],'o')
    # plot the centroids
    lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    pyplot.setp(lines,ms=15.0)
    pyplot.setp(lines,mew=2.0)
    pyplot.title('KMeans with %d clusters' % kmeans_n_clusters)
pyplot.show()
###############################################################################
'''