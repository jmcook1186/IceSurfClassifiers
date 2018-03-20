#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:32:24 2018

@author: joe
"""

# This code provides functions for reading in directional reflectance data obtained
# via ground spectroscopy, reformatting into a pandas dataframe of features and labels,
# optimising and training a series of machine learning algorithms on the ground spectra
# then using the trained model to predict the surface type in each pixel of a Sentinel-2
# image. The Sentinel-2 image has been preprocesses using ESA Sen2Cor command line
# algorithm to convert to surface reflectance before being saved as a multiband TIF which
# is then loaded here. A narrowband to broadband conversion (Knap 1999) is applied to the
# data to create an albedo map, and this is then used to create a large dataset of surface 
# type and associated broadband albedo


###########################################################################################
############################# IMPORT MODULES #########################################

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing, cross_validation, neighbors, svm
import matplotlib.pyplot as plt
import gdal
import rasterio
from sklearn.grid_search import GridSearchCV
plt.style.use('ggplot')


#######################################################################################
############################ DEFINE FUNCTIONS ###################################


def create_dataset():
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
    
    
    # Join dataframes into one continuous DF
    
    X = X.append(Y,ignore_index=True)
    X = X.append(Z,ignore_index=True)
    X = X.append(P,ignore_index=True)
    X = X.append(Q,ignore_index=True)
    X = X.append(Zero,ignore_index=True)    
    
    # Create features and labels (XX = features - all data but no labels, YY = labels only)
    
    XX = X.drop(['label'],1)
    YY = X['label']
    
    return X, XX, YY



def optimise_train_model(X,XX,YY):
    # empty lists to append to
    Naive_Bayes = []
    KNN = []
    SVM = []
    
    # split data into test and train sets
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(XX,YY,test_size = 0.2)

    # Optimize parameters using GridSearch with cross validation (GridSearchCV) to
    # find optimal set of values for best model performance. Apply to three kernel types
    # and wide range of C and gamma values. Print best set of params.  
    
    tuned_parameters = [
            {'kernel': ['linear'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                         'C': [0.1, 1, 10, 100, 1000, 10000]},
                        {'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                         'C': [0.1, 1, 10, 100, 1000, 10000]},
                        {'kernel':['poly'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                         'C':[0.1,1,10,100,1000,10000]},
                        {'kernel':['sigmoid'],'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                         'C':[0.1,1,10,100,1000,10000]}
                        ]
    
    clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5)
    clf.fit(X_train, Y_train)
    
    print()
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print() #line break
    
    kernel = clf.best_estimator_.get_params()['kernel']
    C = clf.best_estimator_.get_params()['C']
    gamma = clf.best_estimator_.get_params()['gamma']
    
    
    # number of times to run train/test with random sample selection - reported
    # accuracy will be mean accuracy for eacxh run
    Num_runs = 100000
    
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(XX,YY,test_size = 0.2)
    
    # Train a range of algorithms and measure accuracy 
    for i in range(1,Num_runs,1): 
                
        # test different classifers
        
        # 1. Try Naive Bayes
        clf = GaussianNB()
        clf.fit(X_train,Y_train)
        accuracy = clf.score(X_test,Y_test)
        Naive_Bayes.append(accuracy)
        
        # 2. Try K-nearest neighbours
        clf = neighbors.KNeighborsClassifier()
        clf.fit(X_train,Y_train)
        accuracy = clf.score(X_test,Y_test)
        KNN.append(accuracy)
        
        # 3. Try support Vector Machine with best params from optimisation
        clf = svm.SVC(kernel=kernel, C=C, gamma = gamma)
        clf.fit(X_train,Y_train)
        accuracy = clf.score(X_test,Y_test)
        SVM.append(accuracy)
    
    
    print('KNN ',np.mean(KNN))
    print('Naive Bayes ', np.mean(Naive_Bayes))
    print('SVM ', np.mean(SVM))
    
    if np.mean(KNN) > np.mean(Naive_Bayes) and np.mean(KNN) > np.mean(SVM):
        clf = clf = neighbors.KNeighborsClassifier()
    elif np.mean(Naive_Bayes) > np.mean(KNN) and np.mean(Naive_Bayes) > np.mean(SVM):
        clf = GaussianNB()
    elif np.mean(SVM) > np.mean(KNN) and np.mean(SVM) > np.mean(Naive_Bayes):
        clf = svm.SVC(kernel=kernel, C=C, gamma = gamma)

    return clf


def ImageAnalysis(clf):

    # Import multispectral imagery from Sentinel 2 and apply ML algorithm to classify surface
    
    jp2s = ['/media/joe/FDB2-2F9B/B02.jp2', '/media/joe/FDB2-2F9B/B03.jp2', '/media/joe/FDB2-2F9B/B04.jp2', '/media/joe/FDB2-2F9B/B08.jp2' ]
    arrs = []
    
    res = 0.01 # Ground resolution of sentinel data in km
    
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
    albedo_array = []
    
    # get dimensions of each band layer
    lenx, leny = np.shape(data[0])
    
    # Loop through each pixel and append the pixel value from each layer to a 1D list
    for i in range(0,lenx,1):
        for j in range(0,leny,1):
            B2.append(data[0][i,j])
            B3.append(data[1][i,j])
            B4.append(data[2][i,j])
            B8.append(data[3][i,j])
    
    # create new array of arrays. Each subarray contains reflectance value for each layer
    # Sen2Cor provides data * 10000, so divide by 10000 to get reflectance between 0-1
    
    # for each pixel, use the B3 and B8 values to apply Knap's narrowband-broadband 
    # conversion to estimate albedo
            
    for i in range(0,len(B2),1):
        test_array.append([ B2[i]/10000, B3[i]/10000, B4[i]/10000, B8[i]/10000 ])
        albedo_array.append(0.726*(B3[i]/10000) - 0.322*(B3[i]/10000)**2 - 0.015*(B8[i]/10000) + 0.581*(B8[i]/10000))
    
    
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
    
    # reshape 1D arrays back into original image dimensions
    predicted = np.reshape(predicted,[lenx,leny])
    albedo_array = np.reshape(albedo_array,[lenx,leny])
    
    # Trim classified image and albedo image to same size, removing non-ice area
    predicted = predicted[2000:-1000,6000:-1]
    albedo_array = albedo_array[2000:-1000,6000:-1]
    
    x,y = np.shape(predicted)
    area_of_pixel = (x*res)*(y*res) # area of selected region
    
    #plot classified surface and albedo
    plt.figure(figsize = (30,30)),plt.imshow(predicted),plt.colorbar()
    plt.savefig('Sentinel2_10m_classified.jpg',dpi=300)
    plt.figure(figsize = (30,30)),plt.imshow(albedo_array),plt.colorbar()
    plt.savefig('Sentinel2_10m_albedo.jpg',dpi=300)
    
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
    LA_coverage = (numLA)/noUNKNOWNS * 100
    CI_coverage = (numCI)/noUNKNOWNS * 100
    CC_coverage = (numCC)/noUNKNOWNS * 100
    WAT_coverage = (numWAT)/noUNKNOWNS * 100
    
    # Print coverage summary
    
    print('**** SUMMARY ****')
    print('Area of pixel = ', area_of_pixel)
    print('% algal coverage (Hbio + Lbio) = ',np.round(tot_alg_coverage,2))
    print('% Hbio coverage = ',np.round(HA_coverage,2))
    print('% Lbio coverage = ',np.round(LA_coverage,2))
    print('% cryoconite coverage = ',np.round(CC_coverage,2))
    print('% clean ice coverage = ',np.round(CI_coverage,2))
    print('% water coverage = ',np.round(WAT_coverage,2))

    return predicted, albedo_array, HA_coverage, LA_coverage, CI_coverage, CC_coverage, WAT_coverage


def albedo_report(predicted,albedo_array):

    alb_WAT = []
    alb_CC = []
    alb_CI = []
    alb_LA = []
    alb_HA = []
    
    predicted = predicted.ravel()
    albedo_array = albedo_array.ravel()
    
    idx_WAT = np.where(predicted ==1)[0]
    idx_CC = np.where(predicted ==2)[0]
    idx_CI = np.where(predicted ==3)[0]
    idx_LA = np.where(predicted ==4)[0]
    idx_HA = np.where(predicted ==5)[0]
    
    for i in idx_WAT:
        alb_WAT.append(albedo_array[i])
    for i in idx_CC:
        alb_CC.append(albedo_array[i])
    for i in idx_CI:
        alb_CI.append(albedo_array[i])
    for i in idx_LA:
        alb_LA.append(albedo_array[i])
    for i in idx_HA:
        alb_HA.append(albedo_array[i])
    
    
    # Calculate summary stats
    mean_CC = np.mean(alb_CC)
    std_CC = np.mean(alb_CC)
    max_CC = np.max(alb_CC)
    min_CC = np.min(alb_CC)

    mean_CI = np.mean(alb_CI)
    std_CI = np.mean(alb_CI)
    max_CI = np.max(alb_CI)
    min_CI = np.min(alb_CI)
    
    mean_LA = np.mean(alb_LA)
    std_LA = np.mean(alb_LA)
    max_LA = np.max(alb_LA)
    min_LA = np.min(alb_LA)

    mean_HA = np.mean(alb_HA)
    std_HA = np.mean(alb_HA)
    max_HA = np.max(alb_HA)
    min_HA = np.min(alb_HA)

    mean_WAT = np.mean(alb_WAT)
    std_WAT = np.mean(alb_WAT)
    max_WAT = np.max(alb_WAT)
    min_WAT = np.min(alb_WAT)
        
    ## FIND IDX WHERE CLASS = Hbio..
    ## BIN ALBEDOS FROM SAME IDXs
    print('mean albedo WAT = ', mean_WAT)
    print('mean albedo CC = ', mean_CC)
    print('mean albedo CI = ', mean_CI)
    print('mean albedo LA = ', mean_LA)
    print('mean albedo HA = ', mean_HA)

    return alb_WAT, alb_CC, alb_CI, alb_LA, alb_HA, mean_CC,std_CC,max_CC,min_CC,mean_CI,std_CI,max_CI,min_CI,mean_LA,mean_HA,std_HA,max_HA,min_HA,mean_WAT,std_WAT,max_WAT,min_WAT

################################################################################
################################################################################



############### RUN ENTIRE SEQUENCE ###################

# create dataset
X,XX,YY = create_dataset()
#optimise and train model
clf = optimise_train_model(X,XX,YY)
# apply model to Sentinel2 image
predicted, albedo_array, HA_coverage, LA_coverage, CI_coverage, CC_coverage, WAT_coverage = ImageAnalysis(clf)
#obtain albedo summary stats
alb_WAT, alb_CC, alb_CI, alb_LA, alb_HA, mean_CC,std_CC,max_CC,min_CC,mean_CI,std_CI,max_CI,min_CI,mean_LA,mean_HA,std_HA,max_HA,min_HA,mean_WAT,std_WAT,max_WAT,min_WAT = albedo_report(predicted,albedo_array)
