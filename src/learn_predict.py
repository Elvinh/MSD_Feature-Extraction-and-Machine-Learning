
# coding: utf-8

# In[1]:

# Loudness analysis with Kmeans clustering
# Created by Elton Vinh and Edward Huang


# In[2]:

import csv

import numpy as np
from numpy import *

from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# In[3]:

# Read the file
csvfile = open('msd_loudness_kmeans_dataset.csv', 'rt')


# In[4]:

lines = csv.reader(csvfile)
dataset = list(lines)
array(dataset).shape #verify it's read correctly


# In[5]:

feature_list = array(dataset[0])
dataset = dataset[1:]
feature_list


# In[6]:

# Splits dataset a more uniform 2:1 split.
def splitN(N):
    #alternate split by doing 2:1
    i = 0
    training_set = []
    test_set = []
    for x in range(10, len(dataset)-1):
        if (i % N == 0):
            test_set.append(array(dataset[x]))
        else:
            training_set.append(array(dataset[x]))
        i += 1
    return(array(training_set), array(test_set))


# In[7]:

# Create the training and test sets using the split function
training_set,test_set = splitN(3)
training_set.shape, test_set.shape


# In[8]:

# Build the input for training
def buildX(training_set, feature_list):
    X = []
    for train in training_set:
        data = []

        for i in feature_list:
            data.append(float(train[i]))

        X.append(array(data))
    X = array(X)
    return X


# In[9]:

# Preprocess output values to determine unique and also re-map to the indices
def buildLabels(y):
    labels = np.unique(y)
    
    y_index = []
    for label in y:
        y_index.append(np.where(labels==label))
    return array(labels), array(y_index)


# In[10]:

# Get Predictions for the test data
def getPredictionsIndex(clf, start, end):
    num_matches = 0
    predictions = []

    for test, index in zip(X_test, y_test_index):
        predicted_ = clf.predict(test)
        predictions.append(predicted_[0])
        if predicted_[0] == index[0][0]:
            num_matches = num_matches + 1
    
    print("Matches: ", num_matches)
    print("Accuracy: ", num_matches/float(end-start))
    return predictions


# In[11]:

# Get the recall score of a genre
def getRecall(label, predictions):
    fn = 0
    tp = 0
    i = 0
    recall = 0
    
    actual = test_set[test_start:test_end]
   
    for prediction in predictions:
        act = actual[i][6]
        
        if prediction != label:
            if act == label:
                fn += 1
        elif act == label:
            tp += 1
        i += 1

    if tp+fn != 0:
        recall = tp/float((tp+fn))
        # print("\nGenre: ", genre)
        # print("True positives: ", tp,"False negatives: ", fn)
        # print("Recall: ", recall)
    return recall


# In[12]:

# Get the precision score of genre
def getPrecision(label, predictions):
    tp = 0
    fp = 0
    i = 0
    precision = 0
    
    actual = test_set[test_start:test_end]

    for pred in predictions:
        act = actual[i][6]
        
        if act == label:
            if pred == act:
                tp += 1
        elif act != label:
            if pred == label:
                fp += 1
        i += 1
    if tp+fp != 0:
        # print( "\nGenre: ", genre)
        # print("True positives: ", tp,"False positives: ", fp)
        precision = tp/float((tp+fp))
        # print("Precision: ", precision)
    return precision


# In[13]:

# Prints tabulated results of the machine learning models
def printResults(predictions):

    print("\n")
    print('%-22s%-6s' % ("Genre/Algo", "KMEANS"))
    for label in test_labels:
        print('%-22s%-.2f' % (label, getPrecision(label, predictions)))
        print('%-22s%-.2f' % ("", getRecall(label, predictions)))


# In[14]:

# Kmeans clustering
def getKmeans(X, y):
    clf = KMeans(n_clusters=3)
    clf.fit(X, y)
    
    return clf


# In[15]:

# define variables for machine learning
feature_list = range(2,5)
test_start = 0
test_end = 3330
train_start = 0
train_end = 6659

X = buildX(training_set, feature_list)
y = training_set[:, 6] # 6 - "Loudness Labels"
labels, y_index = buildLabels(y)
y_index = np.ravel(y_index)

X_test = buildX(test_set, feature_list)
y_test = test_set[:, 6]
test_labels, y_test_index = buildLabels(y_test)


# In[16]:

# Performs Kmeans clustering and prints the statistics of the model. Run more than once f
def cluster_print_results():
    clfKmeans = getKmeans(X, y_index)
    predictionsKmeans_index = getPredictionsIndex(clfKmeans, test_start, test_end)
    predictionsKmeans = []
    for i in predictionsKmeans_index:
        predictionsKmeans.append(test_labels[int(i)])

    printResults(predictionsKmeans)


# In[17]:

# Run more than once for better results
cluster_print_results()


# In[18]:

def show_2d_graph():
    y_pred = KMeans(n_clusters=3).fit_predict(X_test)

    plt.scatter(X_test[:,0], X_test[:,1], c=y_pred)
    plt.title("Clusters")
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test_index)
    plt.title("Input")
    plt.show()


# In[19]:

show_2d_graph()


# In[20]:

def show_3d_graph():
    print(__doc__)

    np.random.seed(5)

    centers = [[1, 1], [-1, -1], [1, -1]]

    X = X_test
    y = y_test_index
    y = np.ravel(y)
    estimators = {'k_means_iris_3': KMeans(n_clusters=3)}

    fignum = 1
    for name, est in estimators.items():
        fig = plt.figure(fignum, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()
        est.fit(X)
        labels = est.labels_

        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels.astype(np.float))

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('loudness')
        ax.set_ylabel('peak_loudness')
        ax.set_zlabel('avg_max_loudness')
        fignum = fignum + 1

    # Plot the ground truth
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()

    for name, label in [('Loud', 0),
                        ('Medium', 1),
                        ('Quiet', 2)]:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Loudness')
    ax.set_ylabel('peak_loudness')
    ax.set_zlabel('avg_max_loudness')
    plt.show()


# In[21]:

show_3d_graph()


# In[22]:

# Kmeans with 2 clusters, attempt at finding clipped vs unclipped
def two_cluster_kmeans():
    # Run the steps.
    feature_list = range(2,6)

    X = buildX(training_set, feature_list)
    y = training_set[:, 6] # Choose 2 for artist name for output, 0 for genre
    labels, y_index = buildLabels(y)
    y_index = np.ravel(y_index)

    X_test = buildX(test_set, feature_list)
    y_test = test_set[:, 6]
    x_labels, y_test_index = buildLabels(y_test)

    clf = KMeans(n_clusters=2)
    clf.fit(X)

    predictions = []
    for test in X_test:
            predicted_ = clf.predict(array(test))
            predictions.append(predicted_[0])

    plt.scatter(X_test[:,0], X_test[:,1], c=predictions)
    plt.title("Clusters")
    plt.show()


# In[23]:

two_cluster_kmeans()


# In[ ]:



