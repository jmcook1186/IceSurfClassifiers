#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:32:24 2018

@author: joe
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing, cross_validation, neighbors, svm
import matplotlib.pyplot as plt
import gdal
import rasterio

plt.style.use('ggplot')


# Read in raw HCRF data to DataFrame. This version pulls in HCRF data from 2016 and 2017

hcrf_master = pd.read_csv('//home//joe//Code//HCRF_master_machine.csv')
HA_hcrf = pd.DataFrame()
LA_hcrf = pd.DataFrame()
CI_hcrf = pd.DataFrame()
CC_hcrf = pd.DataFrame()
WAT_hcrf = pd.DataFrame()

# Group site names

HAsites = ['13_7_SB2','13_7_SB4','14_7_S5','14_7_SB1','14_7_SB5','14_7_SB10',
'15_7_SB3','21_7_SB1','21_7_SB7','22_7_SB4','22_7_SB5','22_7_S3','22_7_S5',
'23_7_SB3','23_7_SB5','23_7_S3','23_7_SB4','24_7_SB2','HA_1', 'HA_2','HA_3',
'HA_4','HA_5','HA_6','HA_7','HA_8','HA_10','HA_11','HA_12','HA_13','HA_14',
'HA_15','HA_16','HA_17','HA_18','HA_19','HA_20','HA_21','HA_22','HA_24',
'HA_25','HA_26','HA_27','HA_28','HA_29','HA_30','HA_31',
# the following were reclassified from LAsites due to their v low reflectance
'13_7_S2','14_7_SB9','MA_11','MA_14','MA_15','MA_17','21_7_SB2','22_7_SB1',
'MA_4','MA_7','MA_18'
]
# These have been removed completely from HAsites: '21_7_S3', '23_7_S5', 'HA_32'
# '24_7_S1','25_7_S1','HA_9', 'HA_33','13_7_SB1', '13_7_S5', 'HA_23'

LAsites = [
'14_7_S2','14_7_S3','14_7_SB2','14_7_SB3','14_7_SB7','15_7_S2','15_7_SB4',
'20_7_SB1','20_7_SB3','21_7_S1','21_7_S5','21_7_SB4','22_7_SB2','22_7_SB3','22_7_S1',
'23_7_S1','23_7_S2','24_7_S2','MA_1','MA_2','MA_3','MA_5','MA_6','MA_8','MA_9',
'MA_10','MA_12','MA_13','MA_16','MA_19',
#these have been moved from CI
'13_7_S1','13_7_S3','14_7_S1','15_7_S1','15_7_SB2','20_7_SB2','21_7_SB5','21_7_SB8','25_7_S3'
]
# These have been removed competely from LA sites
# '13_7_S2','13_7_SB1','14_7_SB9', '15_7_S3' ,'MA_11',' MA_14','MA15','MA_17',
# '13_7_S5', '25_7_S2','25_7_S4','25_7_S5'

CIsites =['13_7_SB3','13_7_SB5','15_7_S4','15_7_SB1','15_7_SB5','21_7_S2',
'21_7_S4','21_7_SB3','22_7_S2','22_7_S4','23_7_SB1','23_7_SB2','23_7_S4',
'WI_1','WI_2','WI_3','WI_4','WI_5','WI_6','WI_7','WI_8','WI_9','WI_10','WI_11',
'WI_12','WI_13']

CCsites = ['DISP1','DISP2','DISP3','DISP4','DISP5','DISP6','DISP7','DISP8',
           'DISP9','DISP10','DISP11','DISP12','DISP13','DISP14']

WATsites = ['21_7_SB5','21_7_SB7','21_7_SB8',
         '25_7_S3', 'WAT_1','WAT_3','WAT_4','WAT_5','WAT_6','WAT_6']

#REMOVED FROM WATER SITES 'WAT_2'

# Create dataframes for ML algorithm
for i in HAsites:
    hcrf_HA = np.array(hcrf_master[i])
    HA_hcrf['{}'.format(i)] = hcrf_HA

for ii in LAsites:
    hcrf_LA = np.array(hcrf_master[ii])
    LA_hcrf['{}'.format(ii)] = hcrf_LA
     
for iii in CIsites:   
    hcrf_CI = np.array(hcrf_master[iii])
    CI_hcrf['{}'.format(iii)] = hcrf_CI   

for iv in CCsites:   
    hcrf_CC = np.array(hcrf_master[iv])
    CC_hcrf['{}'.format(iv)] = hcrf_CC   

for v in WATsites:   
    hcrf_WAT = np.array(hcrf_master[v])
    WAT_hcrf['{}'.format(v)] = hcrf_WAT  

# Make dataframe with column for label, columns for reflectancxe at key wavelengths
# select wavelengths to use - currently set to 8 Sentnel 2 bands

X = pd.DataFrame()

X['R140'] = np.array(HA_hcrf.iloc[140])
X['R210'] = np.array(HA_hcrf.iloc[210])
X['R315'] = np.array(HA_hcrf.iloc[315])
X['R490'] = np.array(HA_hcrf.iloc[490])

X['label'] = 'HA'


Y = pd.DataFrame()
Y['R140'] = np.array(LA_hcrf.iloc[140])
Y['R210'] = np.array(LA_hcrf.iloc[210])
Y['R315'] = np.array(LA_hcrf.iloc[315])
Y['R490'] = np.array(LA_hcrf.iloc[490])

Y['label'] = 'LA'


Z = pd.DataFrame()

Z['R140'] = np.array(CI_hcrf.iloc[140])
Z['R210'] = np.array(CI_hcrf.iloc[210])
Z['R315'] = np.array(CI_hcrf.iloc[315])
Z['R490'] = np.array(CI_hcrf.iloc[490])

Z['label'] = 'CI'


P = pd.DataFrame()

P['R140'] = np.array(CC_hcrf.iloc[140])
P['R210'] = np.array(CC_hcrf.iloc[210])
P['R315'] = np.array(CC_hcrf.iloc[315])
P['R490'] = np.array(CC_hcrf.iloc[490])

P['label'] = 'CC'


Q = pd.DataFrame()
Q['R140'] = np.array(WAT_hcrf.iloc[140])
Q['R210'] = np.array(WAT_hcrf.iloc[210])
Q['R315'] = np.array(WAT_hcrf.iloc[315])
Q['R490'] = np.array(WAT_hcrf.iloc[490])

Q['label'] = 'WAT'


## Zero option added to avoid the classifier assigning a surface type to clipped
# aread of image. Since these areas have zero reflectance values in all bands, 
# the classifier correctly identifies them and assigns the to the 'unknown' category


Zero = pd.DataFrame()
Zero['R140'] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
Zero['R210'] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
Zero['R315'] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
Zero['R490'] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

Zero['label'] = 'UNKNOWN'


# Include Snow as feature ZZ: OPTIONAL (not included by default)

#ZZ = pd.DataFrame()
#ZZ['R125'] = np.array(statsSnow_hcrf.iloc[125])
#ZZ['R210'] = np.array(statsSnow_hcrf.iloc[210])
#ZZ['R318'] = np.array(statsSnow_hcrf.iloc[318])
#ZZ['R367'] = np.array(statsSnow_hcrf.iloc[367])
#ZZ['R490'] = np.array(statsSnow_hcrf.iloc[490])
#
#ZZ['label'] = 'Snow'

# Jin dataframes into one continuous DF

X = X.append(Y,ignore_index=True)
X = X.append(Z,ignore_index=True)
X = X.append(P,ignore_index=True)
X = X.append(Q,ignore_index=True)
X = X.append(Zero,ignore_index=True)
#X = X.append(ZZ,ignore_index=True)




# Create featires and labels (XX = features - all data but no labels, YY = labels only)

XX = X.drop(['label'],1)
YY = X['label']


Naive_Bayes = []
KKN = []
SVM_linear = []
SVM_sigmoid = []
SVM_poly = []
SVM_rbf = []

Num_runs = 10000


# Train a range of algorithms and measure accuracy 

for i in range(1,Num_runs,1):

    # split data into test and train sets
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(XX,YY,test_size = 0.2)
    
    # test different classifers
    
    clf = GaussianNB()
    clf.fit(X_train,Y_train)
    accuracy = clf.score(X_test,Y_test)
    Naive_Bayes.append(accuracy)

    
    # 1. Try K-nearest neighbours
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train,Y_train)
    accuracy = clf.score(X_test,Y_test)
    KKN.append(accuracy)

    
    # 2. Try support Vector Machine (linear kernel)
    clf = svm.SVC(kernel='linear', C=1000.0)
    clf.fit(X_train,Y_train)
    accuracy = clf.score(X_test,Y_test)
    SVM_linear.append(accuracy)



    # 3. Try support Vector Machine (radial basis function kernel)
#    clf = svm.SVC(kernel='rbf', C=1000.0, gamma = 0.1)
#    clf.fit(X_train,Y_train)
#    accuracy = clf.score(X_test,Y_test)
#    SVM_rbf.append(accuracy)
    
    # 4. Try support Vector Machine (polynomial kernel)
#    clf = svm.SVC(kernel='poly', C=1000.0)
#    clf.fit(X_train,Y_train)
#    accuracy = clf.score(X_test,Y_test)
#    SVM_poly.append(accuracy)
#    
#    # 5. Try support Vector Machine (sigmoid kernel)
#    clf = svm.SVC(kernel='sigmoid', C=1000.0)
#    clf.fit(X_train,Y_train)
#    accuracy = clf.score(X_test,Y_test)
#    SVM_sigmoid.append(accuracy)


print('KKN ',np.mean(KKN))
print('Naive Bayes ', np.mean(Naive_Bayes))
print('SVM_linear ', np.mean(SVM_linear))
#print('SVM_sigmoid',np.mean(SVM_sigmoid))
#print('SVM_rbf',np.mean(SVM_rbf))
#print('SVM_poly',np.mean(SVM_poly))


##############################################################################
##################### OPTIMIZING PARAMETERS FOR SVM ##########################

# comment out for actual model fitting, but run to determie optimal params for 
# svm

#
# Optimize parameters using GridSearch with cross validation (GridSearchCV) to
# find optimal set of values for best model performance. Apply to three kernel types
# and wide range of C and gamma values. Print best set of params.
#
#
#from sklearn.grid_search import GridSearchCV
#tuned_parameters = [{'kernel': ['linear'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
#                     'C': [0.1, 1, 10, 100, 1000, 10000]},
#                    {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100, 1000, 10000]},
#                    {'kernel':['poly'], 'C':[0.1,1,10,100,1000,10000]}]
#
#
#clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5)
#clf.fit(X_train, Y_train)
#
#print("Best parameters set found on development set:")
#print()
#print(clf.best_params_)




##############################################################################
####### IMPORT MULTISPECTRAL IMAGE AND CLASSIFY USING ########################
##################   TRAINED ML ALGORITHM   ##################################

# NB THIS SECTION TAKES A LONG TIME TO RUN 
#(>30 mins on laptop - i7 7700 GHz, 32 GB RAM)

# import image from file (importing 4 x jp2 files)

jp2s = ['/media/joe/FDB2-2F9B/B02.jp2', '/media/joe/FDB2-2F9B/B03.jp2', '/media/joe/FDB2-2F9B/B04.jp2', '/media/joe/FDB2-2F9B/B08.jp2' ]
arrs = []

for jp2 in jp2s:
    with rasterio.open(jp2) as f:
        arrs.append(f.read(1))

data = np.array(arrs, dtype=arrs[0].dtype)

# set up empty lists to append into
B2 = []
B3 = []
B4 = []
B8 =[]

predicted = []
test_array = []

# get dimensions of each band layer
lenx, leny = np.shape(data[0])

# Loop through each pixel and append the pixel value from each layer to a 1D list
for i in range(0,lenx,1):
    for j in range(0,leny,1):
        B2.append(data[0][i,j])
        B3.append(data[1][i,j])
        B4.append(data[2][i,j])
        B8.append(data[3][i,j])

# crop image to eliminate non-ice areas


# create new array of arrays. Each subarray contains reflectance value for each layer
# Sen2Cor provides data * 10000, so divide by 10000 to get reflectance between 0-1
        
for i in range(0,len(B2),1):
    test_array.append([ B2[i]/10000, B3[i]/10000, B4[i]/10000, B8[i]/10000 ])

# apply ML algorithm to 4-value array for each pixel - predict surface type
    
predicted = clf.predict(test_array)

# convert surface class (string) to a numeric value for plotting
predicted[predicted == 'UNKNOWN'] = float(0)
predicted[predicted == 'WAT'] = float(1)
predicted[predicted == 'CC'] = float(2)
predicted[predicted == 'CI'] = float(3)
predicted[predicted == 'LA'] = float(4)
predicted[predicted == 'HA'] = float(5)

# ensure array data type is float (required for imshow)
predicted = predicted.astype(float)
# reshape 1D array back into original image dimensions
predicted = np.reshape(predicted,[lenx,leny])
predicted = predicted[2000:-1000,6000:-1]

#plot classified surface
plt.figure(figsize = (30,30)),plt.imshow(predicted),plt.colorbar()

# Calculate coverage stats
numHA = (predicted==5).sum()
numLA = (predicted==4).sum()
numCI = (predicted==3).sum()
numCC = (predicted==2).sum()
numWAT = (predicted==1).sum()
numUNKNOWN = (predicted==0).sum()
noUNKNOWNS = (predicted !=0).sum()

tot_alg_coverage = (numHA+numLA)/noUNKNOWNS *100
HA_coverage = (numHA)/noUNKNOWNS * 100

print('Total Algal Coverage = ', np.round(tot_alg_coverage,2))
###################################




