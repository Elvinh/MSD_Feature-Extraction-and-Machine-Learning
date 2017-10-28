
# coding: utf-8

# In[1]:

# Feature extraction from MSD
# Creates CSV files.
# Created by Elton Vinh


# In[2]:

import csv
import numpy as np
from numpy import *
import os
import glob
import hdf5_getters


# In[3]:

#features extracted: song title, artist, genre, loudness, segments starts, segements loudness max, loudness max
def get_all_titles(basedir,ext='.h5') :
    titles = []
    artist_names = []
    terms = []
    loudness = []
    segments_loudness_max = []
    
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files:
            h5 = hdf5_getters.open_h5_file_read(f)
            
            titles.append(hdf5_getters.get_title(h5)) 
            artist_names.append(hdf5_getters.get_artist_name(h5))
            try:
                terms.append(hdf5_getters.get_artist_terms(h5))
            except:          
                pass
            loudness.append(hdf5_getters.get_loudness(h5))
            try:
                segments_loudness_max.append(hdf5_getters.get_segments_loudness_max(h5))
            except:              
                pass
                        
            h5.close()
    return titles, artist_names, terms, loudness, segments_loudness_max


# In[4]:

# extracts features from MSD, takes a long time
titles, artist_names, terms, loudness, segments_loudness_max = get_all_titles("MillionSongSubset")


# In[5]:

# takes extracted features and writes to csv file
with open('msd_loudness.csv', 'wb') as csvfile:
    csvfile.write("title,artist,loudness,segments_loudness_max\n")
    for title, artist, loud, seg_loud in zip(titles[1:], artist_names[1:], loudness[1:], segments_loudness_max):
        csvfile.write(title + "," + artist + "," + '%f' % (loud))
        for segment in seg_loud:
            csvfile.write("," + '%s' % (segment))
        csvfile.write("\n")       


# In[6]:

# Read the file
csvfile = open('msd_loudness.csv', 'rt')


# In[7]:

lines = csv.reader(csvfile)
dataset1 = list(lines)
array(dataset1).shape #verify it's read correctly


# In[8]:

# Extract feature names
feature_list = array(dataset1[0])
dataset1 = dataset1[1:]
feature_list


# In[9]:

# peak loudness - average loudness
def segment_avg_max_loudness(data_set):
    avg_maxes = []
    
    for track in data_set:
        summ = 0
        number_of_segments = 0
        track_segments = track[3:]
        
        for loudness_max in track_segments:
            if float(loudness_max) > -50:
                summ = summ + float(loudness_max)
                number_of_segments = number_of_segments + 1
        avg_maxes.append(summ/number_of_segments)
    return avg_maxes


# In[10]:

# Max loudness of all segments
def findMaxLoudness(data_set):
    max_loudness = []
    
    for track in data_set:
        maxi = -60.00
        segments = track[3:]
        for peak in segments:
            if(float(maxi) < float(peak)):
                maxi = peak
        max_loudness.append(float(maxi))
        
    return max_loudness


# In[11]:

def label_loudness(data_set):
    loudness_labels = []
    for track in data_set:
        db = float(track[2])
        
        if db > -7:
            loudness_labels.append("loud")
        elif db < -15:
            loudness_labels.append("quiet")
        else:
            loudness_labels.append("medium")
            
    return loudness_labels


# In[12]:

# peak loudness - average loudness
def loudnessDiff(data_set):
    loudnessDiffAvgs = []
    for track in data_set:
        segments = track[3:]
        summ = 0
        diff = 0
        totalSegments = 0
        for segment in segments:
            if (float(segment) > -50):
                diff = float(segment) - float(track[2])
                summ = summ + diff
                totalSegments = totalSegments + 1
        avg = summ/totalSegments
        loudnessDiffAvgs.append(avg)
    return loudnessDiffAvgs


# In[13]:

avg_max_loudness = segment_avg_max_loudness(dataset1)
max_loudness = findMaxLoudness(dataset1)
loudness_labels = label_loudness(dataset1)
loudness_diff_averages = loudnessDiff(dataset1)


# In[14]:

# takes extracted features and writes to csv file
with open('msd_loudness_kmeans_dataset.csv', 'wb') as csvfile:
    csvfile.write("title,artist,loudness,peak_loudness,avg_max_loudness,loudness_diff_averages,loudness_labels\n")
    for track, max_loud, avg_max_loud, loud_diff_avg, label in zip(dataset1, max_loudness, avg_max_loudness, loudness_diff_averages, loudness_labels):
        csvfile.write(track[0] + "," + track[1] + "," + track[2] + "," 
                      + '%.3f,%.3f,%.3f' % (max_loud,avg_max_loud,loud_diff_avg) + "," + label + "\n")


# In[ ]:



