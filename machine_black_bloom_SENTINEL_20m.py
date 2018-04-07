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
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
import gdal
import rasterio
from sklearn.grid_search import GridSearchCV
plt.style.use('ggplot')

HCRF_file = '//home//joe//Code//HCRF_master_machine.csv'
Seninel_jp2s = ['/media/joe/FDB2-2F9B/B02_20m.jp2', '/media/joe/FDB2-2F9B/B03_20m.jp2', '/media/joe/FDB2-2F9B/B04_20m.jp2', '/media/joe/FDB2-2F9B/B05_20m.jp2',
        '/media/joe/FDB2-2F9B/B06_20m.jp2','/media/joe/FDB2-2F9B/B07_20m.jp2','/media/joe/FDB2-2F9B/B8A_20m.jp2',
        '/media/joe/FDB2-2F9B/B11_20m.jp2','/media/joe/FDB2-2F9B/B12_20m.jp2']

#######################################################################################
############################ DEFINE FUNCTIONS ###################################


def create_dataset(HCRF_file):
# Read in raw HCRF data to DataFrame. This version pulls in HCRF data from 2016 and 2017

    hcrf_master = pd.read_csv(HCRF_file)
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
    X['R355'] = np.array(HA_hcrf.iloc[355])
    X['R390'] = np.array(HA_hcrf.iloc[390])
    X['R433'] = np.array(HA_hcrf.iloc[433])
    X['R515'] = np.array(HA_hcrf.iloc[515])
    X['R1260'] = np.array(HA_hcrf.iloc[1260])
    X['R1840'] = np.array(HA_hcrf.iloc[1840])
    
    X['label'] = 'HA'
    
    
    Y = pd.DataFrame()
    Y['R140'] = np.array(LA_hcrf.iloc[140])
    Y['R210'] = np.array(LA_hcrf.iloc[210])
    Y['R315'] = np.array(LA_hcrf.iloc[315])
    Y['R355'] = np.array(LA_hcrf.iloc[355])
    Y['R390'] = np.array(LA_hcrf.iloc[390])
    Y['R433'] = np.array(LA_hcrf.iloc[433])
    Y['R515'] = np.array(LA_hcrf.iloc[515])
    Y['R1260'] = np.array(LA_hcrf.iloc[1260])
    Y['R1840'] = np.array(LA_hcrf.iloc[1840])
    
    Y['label'] = 'LA'
    
    
    Z = pd.DataFrame()
    
    Z['R140'] = np.array(CI_hcrf.iloc[140])
    Z['R210'] = np.array(CI_hcrf.iloc[210])
    Z['R315'] = np.array(CI_hcrf.iloc[315])
    Z['R355'] = np.array(CI_hcrf.iloc[355])
    Z['R390'] = np.array(CI_hcrf.iloc[390])
    Z['R433'] = np.array(CI_hcrf.iloc[433])
    Z['R515'] = np.array(CI_hcrf.iloc[515])
    Z['R1260'] = np.array(CI_hcrf.iloc[1260])
    Z['R1840'] = np.array(CI_hcrf.iloc[1840])
    
    Z['label'] = 'CI'
    
    
    P = pd.DataFrame()
    
    P['R140'] = np.array(CC_hcrf.iloc[140])
    P['R210'] = np.array(CC_hcrf.iloc[210])
    P['R315'] = np.array(CC_hcrf.iloc[315])
    P['R355'] = np.array(CC_hcrf.iloc[355])
    P['R390'] = np.array(CC_hcrf.iloc[390])
    P['R433'] = np.array(CC_hcrf.iloc[433])
    P['R515'] = np.array(CC_hcrf.iloc[515])
    P['R1260'] = np.array(CC_hcrf.iloc[1260])
    P['R1840'] = np.array(CC_hcrf.iloc[1840])
    
    P['label'] = 'CC'
    
    
    Q = pd.DataFrame()
    Q['R140'] = np.array(WAT_hcrf.iloc[140])
    Q['R210'] = np.array(WAT_hcrf.iloc[210])
    Q['R315'] = np.array(WAT_hcrf.iloc[315])
    Q['R355'] = np.array(WAT_hcrf.iloc[355])
    Q['R390'] = np.array(WAT_hcrf.iloc[390])
    Q['R433'] = np.array(WAT_hcrf.iloc[433])
    Q['R515'] = np.array(WAT_hcrf.iloc[515])
    Q['R1260'] = np.array(WAT_hcrf.iloc[1260])
    Q['R1840'] = np.array(WAT_hcrf.iloc[1840])
    
    Q['label'] = 'WAT'
    
    X = X.append(Y,ignore_index=True)
    X = X.append(Z,ignore_index=True)
    X = X.append(P,ignore_index=True)
    X = X.append(Q,ignore_index=True)
    
    # Create features and labels (XX = features - all data but no labels, YY = labels only)
    XX = X.drop(['label'],1)
    YY = X['label']
        
    return X, XX, YY


def optimise_train_model(X,XX,YY, error_selector, test_size = 0.2, plot_all_conf_mx = True):
    
    # Function splits the data into training and test sets, then tests the 
    # performance of a range of models on the training data. The final model 
    # selected is then evaluated using the test set. The function automatically
    # selects the model that performs best on the training sets. All 
    # performance metrics are printed to ipython. The performance metric 
    # to use to select the model is determined in the function call. Options 
    # for error_selector are: 'accuracy', 'F1', 'recall', 'precision', and 
    # 'average_all_metric' 
    # The option 'plot_all_conf_mx' can be se to True or False. If True, the 
    # train set confusion matrices will be plotted for all models. If False,
    # only the final model confusion matrix will be plotted.
    # X, XX, YY are the datasets with and without labels.

    # split data into test and train sets.
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(XX,YY,test_size = test_size)
    
    # test different classifers and report performance metrics using traning data only
    
    # 1. Try Naive Bayes
    clf_NB = GaussianNB()
    clf_NB.fit(X_train,Y_train)
    accuracy_NB = clf_NB.score(X_train,Y_train) #calculate accuracy
    Y_predict_NB = clf_NB.predict(X_train) # make nre prediction
    conf_mx_NB = confusion_matrix(Y_train,Y_predict_NB) # calculate confusion matrix
    recall_NB = recall_score(Y_train,Y_predict_NB,average="weighted")
    f1_NB = f1_score(Y_train, Y_predict_NB, average="weighted") # calculate f1 score
    precision_NB = precision_score(Y_train,Y_predict_NB, average = 'weighted')
    average_metric_NB = (accuracy_NB+recall_NB+f1_NB)/3
    
    # 2. Try K-nearest neighbours
    clf_KNN = neighbors.KNeighborsClassifier()
    clf_KNN.fit(X_train,Y_train)
    accuracy_KNN = clf_KNN.score(X_train,Y_train)
    Y_predict_KNN = clf_KNN.predict(X_train)
    conf_mx_KNN = confusion_matrix(Y_train,Y_predict_KNN)
    recall_KNN = recall_score(Y_train,Y_predict_KNN,average="weighted")
    f1_KNN = f1_score(Y_train, Y_predict_KNN, average="weighted")
    precision_KNN = precision_score(Y_train,Y_predict_KNN, average = 'weighted')
    average_metric_KNN = (accuracy_KNN + recall_KNN + f1_KNN)/3
    
    # 3. Try support Vector Machine with best params calculated using
    # GridSearch cross validation optimisation
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
    
    clf_svm = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=3)
    clf_svm.fit(X_train, Y_train)
    
    print()
    print("Best parameters set found on development set:")
    print(clf_svm.best_params_)
    print() #line break
    
    kernel = clf_svm.best_estimator_.get_params()['kernel']
    C = clf_svm.best_estimator_.get_params()['C']
    gamma = clf_svm.best_estimator_.get_params()['gamma']
    
    clf_svm = svm.SVC(kernel=kernel, C=C, gamma = gamma,probability=True)
    clf_svm.fit(X_train,Y_train)
    accuracy_svm = clf_svm.score(X_train,Y_train)
    Y_predict_svm = clf_svm.predict(X_train)
    conf_mx_svm = confusion_matrix(Y_train,Y_predict_svm)
    recall_svm = recall_score(Y_train,Y_predict_svm,average="weighted")
    f1_svm = f1_score(Y_train, Y_predict_svm, average="weighted")
    precision_svm = precision_score(Y_train,Y_predict_svm, average = 'weighted')
    average_metric_svm = (accuracy_svm + recall_svm + f1_svm)/3

    # 4. Try  a random forest classifier
    clf_RF = RandomForestClassifier(n_estimators = 1000, max_leaf_nodes = 16)
    clf_RF.fit(X_train,Y_train)
    accuracy_RF = clf_RF.score(X_train,Y_train)
    Y_predict_RF = clf_RF.predict(X_train)
    conf_mx_RF = confusion_matrix(Y_train,Y_predict_RF)
    recall_RF = recall_score(Y_train,Y_predict_RF,average="weighted")
    f1_RF = f1_score(Y_train, Y_predict_RF, average="weighted")
    precision_RF = precision_score(Y_train,Y_predict_RF, average = 'weighted')
    average_metric_RF = (accuracy_RF + recall_RF + f1_RF)/3

    # 5. Try an ensemble of all the other classifiers (not RF) using the voting classifier method
    ensemble_clf = VotingClassifier(
            estimators = [('NB',clf_NB),('KNN',clf_KNN),('svm',clf_svm),('RF',clf_RF)],
            voting = 'hard')
    ensemble_clf.fit(X_train,Y_train)
    accuracy_ensemble = ensemble_clf.score(X_train,Y_train)
    Y_predict_ensemble = ensemble_clf.predict(X_train)
    conf_mx_ensemble = confusion_matrix(Y_train,Y_predict_ensemble)
    recall_ensemble = recall_score(Y_train,Y_predict_ensemble,average="weighted")
    f1_ensemble = f1_score(Y_train, Y_predict_ensemble, average="weighted")
    precision_ensemble = precision_score(Y_train,Y_predict_ensemble, average = 'weighted')
    average_metric_ensemble = (accuracy_ensemble + recall_ensemble + f1_ensemble)/3
    
    print()
    print('*** MODEL TEST SUMMARY ***')
    print('KNN accuracy = ',accuracy_KNN, 'KNN_F1_Score = ', f1_KNN, 'KNN Recall = ', recall_KNN, 'KNN precision = ', precision_KNN)
    print('Naive Bayes accuracy = ', accuracy_NB, 'Naive_Bayes_F1_Score = ',f1_NB, 'Naive Bayes Recall = ',recall_NB, 'Naive Bayes Precision = ', precision_NB)
    print('SVM accuracy = ',accuracy_svm, 'SVM_F1_Score = ', f1_svm, 'SVM recall = ', recall_svm, 'SVM Precision = ', precision_svm)
    print('Random Forest accuracy',accuracy_RF,'Random Forest F1 Score = ', f1_RF, 'Random Forest Recall', recall_RF, 'Random Forest Precision = ', precision_RF)    
    print('Ensemble accuracy',accuracy_ensemble,'Ensemble F1 Score = ', f1_ensemble, 'Ensemble Recall', recall_ensemble, 'Ensemble Precision = ', precision_ensemble)
    
    # PLOT CONFUSION MATRICES
    if plot_all_conf_mx:
        
        plt.figure()    
        plt.imshow(conf_mx_NB)
        plt.title('NB Model Confusion matrix')
        plt.colorbar()
        classes = clf_NB.classes_
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        plt.figure()
        plt.imshow(conf_mx_KNN)
        plt.title('KNN Model Confusion matrix')
        plt.colorbar()
        classes = clf_KNN.classes_
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        plt.figure()
        plt.imshow(conf_mx_svm)
        plt.title('SVM Model Confusion matrix')
        plt.colorbar()
        classes = clf_svm.classes_
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        plt.figure()
        plt.imshow(conf_mx_RF)
        plt.title('Random Forest Model Confusion matrix')
        plt.colorbar()
        classes = clf_RF.classes_
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        plt.figure()
        plt.imshow(conf_mx_ensemble)
        plt.title('Ensemble Model Confusion Matrix')
        plt.colorbar()
        classes = ensemble_clf.classes_
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
    print() #line break
    
    if error_selector == 'accuracy':
        
        if accuracy_KNN > accuracy_svm and accuracy_KNN > accuracy_NB and accuracy_KNN > accuracy_RF and accuracy_KNN > accuracy_ensemble:
            clf = neighbours.KNeighboursClassifier()
            clf.fit(X_train,Y_train)
            print('KNN model chosen')
        
        elif accuracy_NB > accuracy_KNN and accuracy_NB > accuracy_svm and accuracy_NB > accuracy_RF and accuracy_NB > accuracy_ensemble:
            clf = clf_NB
            clf.fit(X_train,Y_train)
            print('Naive Bayes model chosen')
            
        elif accuracy_svm > accuracy_NB and accuracy_svm > accuracy_KNN and accuracy_KNN > accuracy_RF and accuracy_svm > accuracy_ensemble:
            clf = clf_svm
            clf.fit(X_train,Y_train)
            print('SVM model chosen')
            print('SVM Params: C = ', C, ' gamma = ', gamma, ' kernel = ', kernel)

        elif accuracy_RF > accuracy_NB and accuracy_RF > accuracy_KNN and accuracy_RF > accuracy_ensemble and accuracy_RF> accuracy_svm:
            clf = clf_RF
            clf.fit(X_train,Y_train)
            print('RF model chosen')
            
        elif accuracy_ensemble > accuracy_svm and accuracy_ensemble > accuracy_NB and accuracy_ensemble > accuracy_RF and accuracy_ensemble > accuracy_KNN:
            clf = clf_ensemble
            clf.fit(X_train,Y_train)
            print('Ensemble model chosen')


    elif error_selector == 'recall':


        if recall_KNN > recall_svm and recall_KNN > recall_NB and recall_KNN > recall_RF and recall_KNN > recall_ensemble:
            clf = neighbours.KNeighboursClassifier()
            clf.fit(X_train,Y_train)
            print('KNN model chosen')
        
        elif recall_NB > recall_KNN and recall_NB > recall_svm and recall_NB > recall_RF and recall_NB > recall_ensemble:
            clf = GaussianNB()
            clf.fit(X_train,Y_train)
            print('Naive Bayes model chosen')
            
        elif recall_svm > recall_NB and recall_svm > recall_KNN and recall_svm > recall_RF and recall_svm > recall_ensemble:
            clf = svm.SVC(kernel=kernel, C=C, gamma = gamma, probability=True)
            clf.fit(X_train,Y_train)
            print('SVM model chosen')
            print('SVM Params: C = ', C, ' gamma = ', gamma, ' kernel = ', kernel)
        
        elif recall_RF > recall_NB and recall_RF > recall_KNN and recall_RF > recall_ensemble and recall_RF> recall_svm:
            clf = clf_RF
            clf.fit(X_train,Y_train)
            print('RF model chosen')


        elif recall_ensemble > recall_svm and recall_ensemble > recall_NB and recall_NB > recall_RF and recall_ensemble > recall_KNN:
            clf = VotingClassifier(
                    estimators = [('NB',clf_NB),('SVM',clf_svm),('KNN',clf_KNN)], voting = 'hard')
            clf.fit(X_train,Y_train)
            print('Ensemble model chosen')

        
    elif error_selector == 'F1':
        
        if f1_KNN > f1_svm and f1_KNN > f1_NB and f1_KNN > f1_RF and f1_KNN > f1_ensemble:
            clf = neighbours.KNeighboursClassifier()
            clf.fit(X_train,Y_train)
            print('KNN model chosen')
        
        elif f1_NB > f1_KNN and f1_NB > f1_svm and f1_NB > f1_RF and f1_NB > f1_ensemble:
            clf = GaussianNB()
            clf.fit(X_train,Y_train)
            print('Naive Bayes model chosen')
            
        elif f1_svm > f1_NB and f1_svm > f1_KNN and f1_svm > f1_RF and f1_svm > f1_ensemble:
            clf = svm.SVC(kernel=kernel, C=C, gamma = gamma, probability = True)
            clf.fit(X_train,Y_train)
            print('SVM model chosen')
            print('SVM Params: C = ', C, ' gamma = ', gamma, ' kernel = ', kernel)
            
        elif f1_RF > f1_NB and f1_RF > f1_KNN and f1_RF > f1_ensemble and f1_RF> f1_svm:
            clf = clf_RF
            clf.fit(X_train,Y_train)
            print('RF model chosen')
        
        elif f1_ensemble > f1_svm and f1_ensemble > f1_NB and f1_ensemble > f1_RF and f1_ensemble > f1_KNN:
            clf = VotingClassifier(
                    estimators = [('NB',clf_NB),('SVM',clf_svm),('KNN',clf_KNN)], voting = 'hard')
            clf.fit(X_train,Y_train)
            print('Ensemble model chosen')
            

    elif error_selector == 'precision':
        
        if precision_KNN > precision_svm and precision_KNN > precision_NB and precision_KNN > precision_RF and precision_KNN > precision_ensemble:
            clf = neighbours.KNeighboursClassifier()
            clf.fit(X_train,Y_train)
            print('KNN model chosen')
        
        elif precision_NB > precision_KNN and precision_NB > precision_svm and precision_NB > precision_RF and precision_NB > precision_ensemble:
            clf = GaussianNB()
            clf.fit(X_train,Y_train)
            print('Naive Bayes model chosen')
            
        elif precision_RF > precision_NB and precision_RF > precision_KNN and precision_RF > precision_ensemble and precision_RF> precision_svm:
            clf = clf_RF
            clf.fit(X_train,Y_train)
            print('RF model chosen')
            
        elif precision_svm > precision_NB and precision_svm > precision_KNN and precision_svm > precision_RF and precision_svm > precision_ensemble:
            clf = svm.SVC(kernel=kernel, C=C, gamma = gamma, probability=True)
            clf.fit(X_train,Y_train)
            print('SVM model chosen')
            print('SVM Params: C = ', C, ' gamma = ', gamma, ' kernel = ', kernel)
        
        elif precision_ensemble > precision_svm and precision_ensemble > precision_NB and precision_ensemble > precision_RF and precision_ensemble > precision_KNN:
            clf = VotingClassifier(
                    estimators = [('NB',clf_NB),('SVM',clf_svm),('KNN',clf_KNN)], voting = 'hard')
            clf.fit(X_train,Y_train)
            print('Ensemble model chosen')

    elif error_selector == 'average_all_metric':
        
        if average_metric_KNN > average_metric_svm and average_metric_KNN > average_metric_NB and average_metric_KNN > average_metric_RF and average_metric_KNN > average_metric_ensemble:
            clf = neighbours.KNeighboursClassifier()
            clf.fit(X_train,Y_train)
            print('KNN model chosen')
        
        elif average_metric_NB > average_metric_KNN and average_metric_NB > average_metric_svm and average_metric_NB > average_metric_RF and average_metric_NB > average_metric_ensemble:
            clf = GaussianNB()
            clf.fit(X_train,Y_train)
            print('Naive Bayes model chosen')
            
        elif average_metric_RF > average_metric_NB and average_metric_RF > average_metric_KNN and average_metric_RF > average_metric_ensemble and average_metric_RF> average_metric_svm:
            clf = clf_RF
            clf.fit(X_train,Y_train)
            print('RF model chosen')
            
        elif average_metric_svm > average_metric_NB and average_metric_svm > average_metric_KNN and average_metric_svm > average_metric_RF and average_metric_svm > average_metric_ensemble:
            clf = svm.SVC(kernel=kernel, C=C, gamma = gamma, probability=True)
            clf.fit(X_train,Y_train)
            print('SVM model chosen')
            print('SVM Params: C = ', C, ' gamma = ', gamma, ' kernel = ', kernel)
        
        elif average_metric_ensemble > average_metric_svm and average_metric_ensemble > average_metric_NB and average_metric_ensemble > average_metric_RF and average_metric_ensemble > average_metric_KNN:
            clf = VotingClassifier(
                    estimators = [('NB',clf_NB),('SVM',clf_svm),('KNN',clf_KNN)], voting = 'hard')
            clf.fit(X_train,Y_train)
            print('Ensemble model chosen')
# Now that model has been selected using error metrics from training data, the final
# model can be evaluated on the test set. The code below therefore measures the f1, recall,
# confusion matrix and accuracy  for the final selected model and prints to ipython.
            
    Y_test_predicted = clf.predict(X_test)
    final_conf_mx = confusion_matrix(Y_test, Y_test_predicted)

    plt.figure()
    plt.imshow(final_conf_mx)
    plt.title('Final Model Confusion Matrix')
    plt.colorbar()
    classes = clf.classes_
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Normalise confusion matrix to show errors
    row_sums = final_conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = final_conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    plt.figure()
    plt.imshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.colorbar()
    plt.title('Normalised Confusion Matrix')
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    final_recall = recall_score(Y_test,Y_test_predicted,average="weighted")
    final_f1 = f1_score(Y_test, Y_test_predicted, average="weighted")
    final_accuracy = clf.score(X_test,Y_test)
    final_precision = precision_score(Y_test, Y_test_predicted, average='weighted')
    final_average_metric = (final_recall + final_accuracy + final_f1)/3

    print() #line break
    print ('*** FINAL MODEL SUMMARY ***')
    print('Final Model Accuracy = ', final_accuracy)
    print('Final Model Recall = ', final_recall)
    print('Final Model F1 = ', final_f1)
    print('Final Model Precision = ',final_precision)
    print('Final Model Average metric = ', final_average_metric)


    return clf


def ImageAnalysis(Sentinel_jp2s,clf):

    # Import multispectral imagery from Sentinel 2 and apply ML algorithm to classify surface
    
    jp2s = Seninel_jp2s
    arrs = []
    
    for jp2 in jp2s:
        with rasterio.open(jp2) as f:
            arrs.append(f.read(1))
    
    res = 0.02 # Ground resolution of sentinel data in km
    data = np.array(arrs, dtype=arrs[0].dtype)

    # set up empty lists to append into
    B2 = []
    B3 = []
    B4 = []
    B5 = []
    B6 = []
    B7 = []
    B8 =[]
    B11 = []
    B12 = []
    
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
            B5.append(data[3][i,j])
            B6.append(data[4][i,j])
            B7.append(data[5][i,j])
            B8.append(data[6][i,j])
            B11.append(data[7][i,j])
            B12.append(data[8][i,j])
    
    # crop image to eliminate non-ice areas
    
    
    # create new array of arrays. Each subarray contains reflectance value for each layer
    # Sen2Cor provides data * 10000, so divide by 10000 to get reflectance between 0-1
    # Apply Liang et al (2002) narrowband to broadband conversion 
            
    for i in range(0,len(B2),1):
        test_array.append([ B2[i]/10000, B3[i]/10000, B4[i]/10000, B5[i]/10000, B6[i]/10000,
                           B7[i]/10000,B8[i]/10000,B11[i]/10000,B12[i]/10000],)
        albedo_array.append(0.356*(B2[i]/10000)+0.130*(B4[i]/10000)+0.373*(B8[i]/10000)+0.085*(B11[i]/10000)+0.072*(B12[i]/10000)-0.0018)
                
    
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
    albedo_array = np.reshape(albedo_array,[lenx,leny])
    predicted = predicted[1000:-1000,3000:]
    albedo_array = albedo_array[1000:-1000,3000:]
    
    x,y = np.shape(predicted)
    area_of_pixel = (x*res)*(y*res) # area of selected region
    #plot classified surface
    plt.figure(figsize = (30,30)),plt.imshow(predicted),plt.colorbar()
    plt.savefig('Sentinel2_20m_classified.jpg',dpi=300)
    plt.figure(figsize = (30,30)),plt.imshow(albedo_array),plt.colorbar()
    plt.savefig('Sentinel2_20m_albedo.jpg',dpi=300)
    # Calculate coverage stats
    numHA = (predicted==5).sum()
    numLA = (predicted==4).sum()
    numCI = (predicted==3).sum()
    numCC = (predicted==2).sum()
    numWAT = (predicted==1).sum()
    numUNKNOWN = (predicted==0).sum()
    noUNKNOWNS = (predicted !=0).sum()
    
    tot_alg_coverage = (numHA+numLA)/noUNKNOWNS*100
    HA_coverage = (numHA)/noUNKNOWNS * 100
    LA_coverage = (numLA)/noUNKNOWNS * 100
    CI_coverage = (numCI)/noUNKNOWNS * 100
    CC_coverage = (numCC)/noUNKNOWNS * 100
    WAT_coverage = (numWAT)/noUNKNOWNS * 100
    
    # Print coverage summary
    
    print('**** SUMMARY ****')
    print('Area of pixel = ', area_of_pixel, 'km')
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
    std_CC = np.std(alb_CC)
    max_CC = np.max(alb_CC)
    min_CC = np.min(alb_CC)

    mean_CI = np.mean(alb_CI)
    std_CI = np.std(alb_CI)
    max_CI = np.max(alb_CI)
    min_CI = np.min(alb_CI)
    
    mean_LA = np.mean(alb_LA)
    std_LA = np.std(alb_LA)
    max_LA = np.max(alb_LA)
    min_LA = np.min(alb_LA)

    mean_HA = np.mean(alb_HA)
    std_HA = np.std(alb_HA)
    max_HA = np.max(alb_HA)
    min_HA = np.min(alb_HA)

    mean_WAT = np.mean(alb_WAT)
    std_WAT = np.std(alb_WAT)
    max_WAT = np.max(alb_WAT)
    min_WAT = np.min(alb_WAT)
        
    ## FIND IDX WHERE CLASS = Hbio..
    ## BIN ALBEDOS FROM SAME IDXs
    print('mean albedo WAT = ', mean_WAT)
    print('mean albedo CC = ', mean_CC)
    print('mean albedo CI = ', mean_CI)
    print('mean albedo LA = ', mean_LA)
    print('mean albedo HA = ', mean_HA)

    albedo_DF = pd.DataFrame(columns=['albedo','class'])
    albedo_DF['class'] = predicted
    albedo_DF['albedo'] = albedo_array
    albedo_DF.to_csv('Sentinel_albedo_20m_dataset.csv')

    return alb_WAT, alb_CC, alb_CI, alb_LA, alb_HA, mean_CC,std_CC,max_CC,min_CC,mean_CI,std_CI,max_CI,min_CI,mean_LA,std_LA,max_LA,min_LA,mean_HA,std_HA,max_HA,min_HA,mean_WAT,std_WAT,max_WAT,min_WAT

################################################################################
################################################################################



############### RUN ENTIRE SEQUENCE ###################

# create dataset
X,XX,YY = create_dataset(HCRF_file)
#optimise and train model
clf =  optimise_train_model(X,XX,YY, error_selector = 'precision', test_size = 0.3, plot_all_conf_mx = False)
# apply model to Sentinel2 image
predicted, albedo_array, HA_coverage, LA_coverage, CI_coverage, CC_coverage, WAT_coverage = ImageAnalysis(Seninel_jp2s,clf)
#obtain albedo summary stats
alb_WAT, alb_CC, alb_CI, alb_LA, alb_HA, mean_CC,std_CC,max_CC,min_CC,mean_CI,std_CI,max_CI,min_CI,mean_LA,std_LA,max_LA,min_LA,mean_HA,std_HA,max_HA,min_HA,mean_WAT,std_WAT,max_WAT,min_WAT = albedo_report(predicted,albedo_array)
