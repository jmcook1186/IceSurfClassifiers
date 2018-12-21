#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 17:50:35 2018

@author: joe
"""

############################# OVERVIEW #######################################

# This code trains a range of supevised classification algorithms on multispectral
# data obtained by reducing hyperspectral data from field spectroscopy down to
# five key wavelengths matching those measured by the MicaSense Red-Edge camera.

# The best performing model is then applied to multispectral imagery obtained
# using the red-edge camera mounted to a UAV. The algorithm classifies each
# pixel according to a function of its reflectance in the five wavelengths.
# These classified pixels are then mapped and the spatial statistics reported.

# A narrowband-broadband coversion function is then used to estimate the albedo
# of each classified pixel, generating a dataset of surface type and albedo.

############################# DETAIL #########################################

# This code is divided into several functions. The first preprocesses the raw data
# into a format appropriate for machine learning. The raw hyperspectral data is
# first organised into separate pandas dataframes for each surface class.
# The data is then reduced down to the reflectance at the five key wavelengths
# and the remaining data discarded. The dataset is then arranged into columns
# with one column per wavelength and a separate column for the surface class.
# The dataset's features are the reflectance at each wavelength, and the labels
# are the surface types. The dataframes for each surface type are merged into
# one large dataframe and then the labels are removed and saved as a separate
# dataframe. XX contains all the data features, YY contains the labels only. No
# scaling of the data is required because the reflectance is already normalised
# between 0 and 1 by the spectrometer.

# The UAV image has been preprocessed in Agisoft Photoscan, including stitching
# and conversion of raw DN to reflectance using calibrated reflectance panels
# on the ground.

# The second function trains a series of supervised classification algorithms.
# The dataset is first divided into a train set and test set at a ratio defined
# by the user (default = 80% train, 20% test). A suite of individual classifiers
# plus two ensemble models are used:
# Individual models are SVM (optimised using GridSearchCV with C between
# 0.0001 - 100000 and gamma between 0.0001 and 1, rbf, polynomial and
# linear kernels), Naive Bayes, K-Nearest Neighbours. Ensemble models are a voting
# classifier (incorporating all the individual models) and a random forest.

# Each classifier is trained and the performance on the training set is reported.
# The user can define which performance measure is most important, and the
# best performing classifier according to the chosen metric is automatically
# selected as the final model. That model is then evaluated on the test set
# and used to classify each pixel in the UAV image. The classified image is
# displayed and the spatial statistics calculated.
#
# NB The classifier can also be loaded in from a joblib save file - in this case
# omit the call to the optimise_train_model() function and simply load the
# trained classifier into the workspace with the variable name 'clf'. Run the other
# functions as normal.

# The trained classifier can also be exported to a joblib savefile by running the
# save_classifier() function,enabling the trained model to be replicated in other
# scripts.

# The albedo of each classified pixel is then calculated from the reflectance
# at each individual wavelength using the narrowband-broadband conversion of
# Knap (1999), creating a final dataframe containing broadband albedo and
# surface type.

##############################################################################
####################### IMPORT MODULES #######################################

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import gdal
from datetime import datetime

from dask_ml.wrappers import ParallelPostFit
plt.style.use('ggplot')

# Set path to csv file containing hcrf spectra (training data) and UAV image to
# be classified

HCRF_file = '//home//joe//Code//HCRF_master_machine_snicar.csv'
img_name = '//home//joe//Desktop//Machine_Learn_Tutorial//UAV_21_7_17//uav_21_7_5cm_commongrid.tif'
savefig_path = '//home/joe/Desktop/'


###############################################################################
########################## DEFINE FUNCTIONS ###################################


def create_dataset(HCRF_file, plot_spectra=True, savefigs=True):
    # Read in raw HCRF data to DataFrame. Pulls in HCRF data from 2016 and 2017

    hcrf_master = pd.read_csv(HCRF_file)
    HA_hcrf = pd.DataFrame()
    LA_hcrf = pd.DataFrame()
    CI_hcrf = pd.DataFrame()
    CC_hcrf = pd.DataFrame()
    WAT_hcrf = pd.DataFrame()
    SN_hcrf = pd.DataFrame()

    # Group data according to site names
    HAsites = ['13_7_SB2', '13_7_SB4', '14_7_S5', '14_7_SB1', '14_7_SB5', '14_7_SB10',
               '15_7_SB3', '21_7_SB1', '21_7_SB7', '22_7_SB4', '22_7_SB5', '22_7_S3', '22_7_S5',
               '23_7_SB3', '23_7_SB5', '23_7_S3', '23_7_SB4', '24_7_SB2', 'HA_1', 'HA_2', 'HA_3',
               'HA_4', 'HA_5', 'HA_6', 'HA_7', 'HA_8', 'HA_10', 'HA_11', 'HA_12', 'HA_13', 'HA_14',
               'HA_15', 'HA_16', 'HA_17', 'HA_18', 'HA_19', 'HA_20', 'HA_21', 'HA_22', 'HA_24',
               'HA_25', 'HA_26', 'HA_27', 'HA_28', 'HA_29', 'HA_30', 'HA_31',
               # the following were reclassified from LAsites due to their v low reflectance
               '13_7_S2', '14_7_SB9', 'MA_11', 'MA_14', 'MA_15', 'MA_17', '21_7_SB2', '22_7_SB1',
               'MA_4', 'MA_7', 'MA_18',
               '27_7_16_SITE3_WMELON1', '27_7_16_SITE3_WMELON3', '27_7_16_SITE2_ALG1',
               '27_7_16_SITE2_ALG2', '27_7_16_SITE2_ALG3', '27_7_16_SITE2_ICE3', '27_7_16_SITE2_ICE5',
               '27_7_16_SITE3_ALG4', '5_8_16_site2_ice7', '5_8_16_site3_ice2', '5_8_16_site3_ice3',
               '5_8_16_site3_ice5', '5_8_16_site3_ice6', '5_8_16_site3_ice7',
               '5_8_16_site3_ice8', '5_8_16_site3_ice9'
               ]
    # These have been removed completely from HAsites: '21_7_S3', '23_7_S5', 'HA_32'
    # '24_7_S1','25_7_S1','HA_9', 'HA_33','13_7_SB1', '13_7_S5', 'HA_23'

    LAsites = [
        '14_7_S2', '14_7_S3', '14_7_SB2', '14_7_SB3', '14_7_SB7', '15_7_S2', '15_7_SB4',
        '20_7_SB1', '20_7_SB3', '21_7_S1', '21_7_S5', '21_7_SB4', '22_7_SB2', '22_7_SB3', '22_7_S1',
        '23_7_S1', '23_7_S2', '24_7_S2', 'MA_1', 'MA_2', 'MA_3', 'MA_5', 'MA_6', 'MA_8', 'MA_9',
        'MA_10', 'MA_12', 'MA_13', 'MA_16', 'MA_19',
        # these have been moved from CI
        '13_7_S1', '13_7_S3', '14_7_S1', '15_7_S1', '15_7_SB2', '20_7_SB2', '21_7_SB5', '21_7_SB8', '25_7_S3',
        '5_8_16_site2_ice10', '5_8_16_site2_ice5', '5_8_16_site2_ice9', '27_7_16_SITE3_WHITE3'
    ]
    # ambiguous spectra removed
    # '13_7_S2','13_7_SB1','14_7_SB9', '15_7_S3' ,'MA_11',' MA_14','MA15','MA_17',
    # '13_7_S5', '25_7_S2','25_7_S4','25_7_S5'

    CIsites = ['21_7_S4', '13_7_SB3', '15_7_S4', '15_7_SB1', '15_7_SB5', '21_7_S2',
               '21_7_SB3', '22_7_S2', '22_7_S4', '23_7_SB1', '23_7_SB2', '23_7_S4',
               'WI_1', 'WI_2', 'WI_4', 'WI_5', 'WI_6', 'WI_7', 'WI_9', 'WI_10', 'WI_11',
               'WI_12', 'WI_13', '27_7_16_SITE3_WHITE1', '27_7_16_SITE3_WHITE2',
               '27_7_16_SITE2_ICE2', '27_7_16_SITE2_ICE4', '27_7_16_SITE2_ICE6',
               '5_8_16_site2_ice1', '5_8_16_site2_ice2', '5_8_16_site2_ice3', '5_8_16_site2_ice4',
               '5_8_16_site2_ice6', '5_8_16_site2_ice8', '5_8_16_site3_ice1', '5_8_16_site3_ice4'
               ]  # ambiguous spectra removed: '13_7_SB5', WI_3, WI_8

    CCsites = ['DISP1', 'DISP2', 'DISP3', 'DISP4', 'DISP5', 'DISP6', 'DISP7', 'DISP8',
               'DISP9', 'DISP10', 'DISP11', 'DISP12', 'DISP13', 'DISP14', '27_7_16_SITE3_DISP1', '27_7_16_SITE3_DISP3',
               ]

    WATsites = ['21_7_SB5', '21_7_SB8', 'WAT_1', 'WAT_3', 'WAT_6']
    # REMOVED FROM WATER SITES 'WAT_2','WAT_4','WAT_5'

    SNsites = ['14_7_S4', '14_7_SB6', '14_7_SB8', '17_7_SB2', 'SNICAR100', 'SNICAR200',
               'SNICAR300', 'SNICAR400', 'SNICAR500', 'SNICAR600', 'SNICAR700', 'SNICAR800', 'SNICAR900', 'SNICAR1000',
               '27_7_16_KANU_', '27_7_16_SITE2_1', '5_8_16_site1_snow10', '5_8_16_site1_snow2', '5_8_16_site1_snow3',
               '5_8_16_site1_snow4', '5_8_16_site1_snow6',
               '5_8_16_site1_snow7', '5_8_16_site1_snow9']

    # REMOVED FROM WATER SITES 'WAT_2'

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

    for vi in SNsites:
        hcrf_SN = np.array(hcrf_master[vi])
        SN_hcrf['{}'.format(vi)] = hcrf_SN
        # Make dataframe with column for label, columns for reflectancxe at key wavelengths

    if plot_spectra:
        WL = np.arange(350, 2501, 1)

        # Creates two subplots and unpacks the output array immediately
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(321)
        ax1.plot(WL, HA_hcrf), plt.xlim(350, 2000), plt.ylim(0, 1.2), plt.title('HA'), plt.xlabel(
            'Wavelength (nm)'), plt.ylabel('HCRF')
        ax1.set_title('Hbio')
        ax2 = fig.add_subplot(322)
        ax2.plot(WL, LA_hcrf), plt.xlim(350, 2000), plt.ylim(0, 1.2), plt.title('LA'), plt.xlabel(
            'Wavelength (nm)'), plt.ylabel('HCRF')
        ax2.set_title('Lbio')
        ax3 = fig.add_subplot(323)
        ax3.plot(WL, CI_hcrf), plt.xlim(350, 2000), plt.ylim(0, 1.2), plt.title('LA'), plt.xlabel(
            'Wavelength (nm)'), plt.ylabel('HCRF')
        ax3.set_title('Clean ice')
        ax4 = fig.add_subplot(324)
        ax4.plot(WL, CC_hcrf), plt.xlim(350, 2000), plt.ylim(0, 1.2), plt.title('LA'), plt.xlabel(
            'Wavelength (nm)'), plt.ylabel('HCRF')
        ax4.set_title('Cryoconite')
        ax5 = fig.add_subplot(325)
        ax5.plot(WL, WAT_hcrf), plt.xlim(350, 2000), plt.ylim(0, 1.2), plt.title('LA'), plt.xlabel(
            'Wavelength (nm)'), plt.ylabel('HCRF')
        ax5.set_title('Water')
        ax6 = fig.add_subplot(326)
        ax6.plot(WL, SN_hcrf), plt.xlim(350, 2000), plt.ylim(0, 1.2), plt.title('LA'), plt.xlabel(
            'Wavelength (nm)'), plt.ylabel('HCRF')
        ax6.set_title('Snow')
        plt.tight_layout()

        if savefigs:
            plt.savefig(str(savefig_path + "training_spectra.jpg"))

        plt.show()

    X = pd.DataFrame()

    X['R475'] = np.array(HA_hcrf.iloc[125])
    X['R560'] = np.array(HA_hcrf.iloc[210])
    X['R668'] = np.array(HA_hcrf.iloc[318])
    X['R717'] = np.array(HA_hcrf.iloc[367])
    X['R840'] = np.array(HA_hcrf.iloc[490])

    X['label'] = 'HA'

    Y = pd.DataFrame()
    Y['R475'] = np.array(LA_hcrf.iloc[125])
    Y['R560'] = np.array(LA_hcrf.iloc[210])
    Y['R668'] = np.array(LA_hcrf.iloc[318])
    Y['R717'] = np.array(LA_hcrf.iloc[367])
    Y['R840'] = np.array(LA_hcrf.iloc[490])

    Y['label'] = 'LA'

    Z = pd.DataFrame()

    Z['R475'] = np.array(CI_hcrf.iloc[125])
    Z['R560'] = np.array(CI_hcrf.iloc[210])
    Z['R668'] = np.array(CI_hcrf.iloc[318])
    Z['R717'] = np.array(CI_hcrf.iloc[367])
    Z['R840'] = np.array(CI_hcrf.iloc[490])

    Z['label'] = 'CI'

    P = pd.DataFrame()

    P['R475'] = np.array(CC_hcrf.iloc[125])
    P['R560'] = np.array(CC_hcrf.iloc[210])
    P['R668'] = np.array(CC_hcrf.iloc[318])
    P['R717'] = np.array(CC_hcrf.iloc[367])
    P['R840'] = np.array(CC_hcrf.iloc[490])

    P['label'] = 'CC'

    Q = pd.DataFrame()
    Q['R475'] = np.array(WAT_hcrf.iloc[125])
    Q['R560'] = np.array(WAT_hcrf.iloc[210])
    Q['R668'] = np.array(WAT_hcrf.iloc[318])
    Q['R717'] = np.array(WAT_hcrf.iloc[367])
    Q['R840'] = np.array(WAT_hcrf.iloc[490])

    Q['label'] = 'WAT'

    R = pd.DataFrame()
    R['R475'] = np.array(SN_hcrf.iloc[125])
    R['R560'] = np.array(SN_hcrf.iloc[210])
    R['R668'] = np.array(SN_hcrf.iloc[318])
    R['R717'] = np.array(SN_hcrf.iloc[367])
    R['R840'] = np.array(SN_hcrf.iloc[490])

    R['label'] = 'SN'

    Zero = pd.DataFrame()
    Zero['R475'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R560'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R668'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R717'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R840'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    Zero['label'] = 'UNKNOWN'

    # Join dataframes into one continuous DF
    X = X.append(Y, ignore_index=True)
    X = X.append(Z, ignore_index=True)
    X = X.append(P, ignore_index=True)
    X = X.append(Q, ignore_index=True)
    X = X.append(R, ignore_index=True)
    X = X.append(Zero, ignore_index=True)

    # Create features and labels (XX = features - all data but no labels, YY = labels only)
    XX = X.drop(['label'], 1)
    YY = X['label']

    return X, XX, YY


def optimise_train_model(X, XX, YY, test_size=0.3):

    # Function splits the data into training and test sets, then applies a Random Forest model
    # dask is used so that the model's predict function can be parallelised later

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(XX, YY, test_size=test_size)

    clf = ParallelPostFit(RandomForestClassifier(n_estimators=1000, max_leaf_nodes=16, n_jobs=-1))

    clf.fit(X_train, Y_train)

    accuracy_RF = clf.score(X_train, Y_train)
    Y_predict_RF = clf.predict(X_train)
    conf_mx_RF = confusion_matrix(Y_train, Y_predict_RF)
    recall_RF = recall_score(Y_train, Y_predict_RF, average="weighted")
    f1_RF = f1_score(Y_train, Y_predict_RF, average="weighted")
    precision_RF = precision_score(Y_train, Y_predict_RF, average='weighted')
    average_metric_RF = (accuracy_RF + recall_RF + f1_RF) / 3


    print("\nModel Performance","\n","\nRandom Forest accuracy = ", accuracy_RF, "\nRandom Forest F1 Score = ", f1_RF, "\nRandom Forest Recall = ",
          recall_RF, "\nRandom Forest Precision = ", precision_RF, "\naverage of all metrics = ", average_metric_RF)


    return clf



def ImageAnalysis(img_name, clf, plot_maps = True, savefigs=True):

    startTime = datetime.now()
    # set up empty lists to append into
    predicted = []
    test_array = []
    arrays = []
    albedo_array = []

    # use gdal to open image and assign each layer to a separate numpy array
    ds = gdal.Open(img_name, gdal.GA_ReadOnly)
    for i in range(1, ds.RasterCount + 1):
        arrays.append(ds.GetRasterBand(i).ReadAsArray())

    # get the length and width of the image from numpy.shape
    lenx, leny = np.shape(arrays[0])

    # convert image bands into single 5-dimensional numpy array
    test_array = np.array([arrays[0] - 0.17, arrays[1] - 0.18, arrays[2] - 0.15, arrays[3] - 0.16, arrays[4] - 0.1])
    test_array = test_array.reshape(5, lenx * leny)  # reshape into 5 x 1D arrays
    test_array = test_array.transpose()  # transpose so that bands are read as features
    # create albedo array by applying Knap (1999) narrowband - broadband conversion
    albedo_array = np.array([0.726 * (arrays[1] - 0.18) - 0.322 * (arrays[1] - 0.18) ** 2 - 0.015 * (
                arrays[3] - 0.16) + 0.581 * (arrays[3] - 0.16)])

    # apply classifier to each pixel in multispectral image with bands as features
    predicted = clf.predict(test_array)

    # convert surface class (string) to a numeric value for plotting
    predicted[predicted == 'UNKNOWN'] = float(0)
    predicted[predicted == 'SN'] = float(1)
    predicted[predicted == 'WAT'] = float(2)
    predicted[predicted == 'CC'] = float(3)
    predicted[predicted == 'CI'] = float(4)
    predicted[predicted == 'LA'] = float(5)
    predicted[predicted == 'HA'] = float(6)

    predicted = predicted.astype(float)

    # reshape 1D array back into original image dimensions
    predicted = np.reshape(predicted, [lenx, leny])
    albedo_array = albedo_array.reshape(lenx, leny)

    # set color scheme for plots - custom for predicted
    cmap1 = mpl.colors.ListedColormap(['white', 'white', 'slategray', 'black', 'lightsteelblue', 'gold', 'orangered'])
    cmap1.set_under(color='white')  # make sure background is white
    cmap2 = plt.get_cmap('Greys_r')  # reverse greyscale for albedo
    cmap2.set_under(color='white')  # make sure background is white

    # plots


    if plot_maps:

        plt.figure(figsize=(15, 15))
        plt.title("Classified ice surface and its albedos from UAV imagery: SW Greenland Ice Sheet", fontsize=28)

        plt.subplot(211)
        plt.imshow(predicted, cmap=cmap1), plt.grid(None), plt.colorbar(), plt.title("UAV Classified Map")

        plt.subplot(212)
        plt.imshow(albedo_array, cmap=cmap2, vmin=0.00000001, vmax = 1), plt.grid(None), plt.colorbar(),\
        plt.title("UAV Albedo Map")

        if not savefigs:
            plt.show()

    if savefigs:
        plt.savefig(str(savefig_path + "UAV_classified_albedo_map.jpg"), dpi=300)
        plt.show()


    print("\nTime taken to classify image = ", datetime.now() - startTime)


    # Calculate coverage stats
    numHA = (predicted == 6).sum()
    numLA = (predicted == 5).sum()
    numCI = (predicted == 4).sum()
    numCC = (predicted == 3).sum()
    numWAT = (predicted == 2).sum()
    numSN = (predicted == 1).sum()
    noUNKNOWNS = (predicted != 0).sum()

    tot_alg_coverage = (numHA + numLA) / noUNKNOWNS * 100
    HA_coverage = (numHA) / noUNKNOWNS * 100
    LA_coverage = (numLA) / noUNKNOWNS * 100
    CI_coverage = (numCI) / noUNKNOWNS * 100
    CC_coverage = (numCC) / noUNKNOWNS * 100
    WAT_coverage = (numWAT) / noUNKNOWNS * 100
    SN_coverage = (numSN) / noUNKNOWNS * 100
    # Print coverage summary

    print('**** SUMMARY ****')
    print('% algal coverage (Hbio + Lbio) = ', np.round(tot_alg_coverage, 2))
    print('% Hbio coverage = ', np.round(HA_coverage, 2))
    print('% Lbio coverage = ', np.round(LA_coverage, 2))
    print('% cryoconite coverage = ', np.round(CC_coverage, 2))
    print('% clean ice coverage = ', np.round(CI_coverage, 2))
    print('% water coverage = ', np.round(WAT_coverage, 2))
    print('% snow coverage', np.round(SN_coverage, 2))
    print('Time taken to classify image = ', datetime.now() - startTime)

    return predicted, albedo_array, HA_coverage, LA_coverage, CI_coverage, CC_coverage, WAT_coverage, SN_coverage


def albedo_report(predicted, albedo_array):

    alb_WAT = []
    alb_CC = []
    alb_CI = []
    alb_LA = []
    alb_HA = []
    alb_SN = []

    predicted = np.array(predicted).ravel()
    albedo_array = np.array(albedo_array).ravel()

    idx_SN = np.where(predicted == 1)[0]
    idx_WAT = np.where(predicted == 2)[0]
    idx_CC = np.where(predicted == 3)[0]
    idx_CI = np.where(predicted == 4)[0]
    idx_LA = np.where(predicted == 5)[0]
    idx_HA = np.where(predicted == 6)[0]

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
    for i in idx_SN:
        alb_SN.append(albedo_array[i])

    # create pandas dataframe containing albedo data (delete rows where albedo <= 0)
    albedo_DF = pd.DataFrame(columns=['albedo', 'class'])
    albedo_DF['class'] = predicted
    albedo_DF['albedo'] = albedo_array
    albedo_DF = albedo_DF[albedo_DF['albedo'] > 0]
    albedo_DF.to_csv('UAV_albedo_dataset.csv')

    # divide albedo dataframe into individual classes for summary stats. include only
    # rows where albedo is between 0.05 and 0.95 percentiles to remove outliers

    HA_DF = albedo_DF[albedo_DF['class'] == 6]
    HA_DF = HA_DF[HA_DF['albedo'] > HA_DF['albedo'].quantile(0.05)]
    HA_DF = HA_DF[HA_DF['albedo'] < HA_DF['albedo'].quantile(0.95)]

    LA_DF = albedo_DF[albedo_DF['class'] == 5]
    LA_DF = LA_DF[LA_DF['albedo'] > LA_DF['albedo'].quantile(0.05)]
    LA_DF = LA_DF[LA_DF['albedo'] < LA_DF['albedo'].quantile(0.95)]

    CI_DF = albedo_DF[albedo_DF['class'] == 4]
    CI_DF = CI_DF[CI_DF['albedo'] > CI_DF['albedo'].quantile(0.05)]
    CI_DF = CI_DF[CI_DF['albedo'] < CI_DF['albedo'].quantile(0.95)]

    CC_DF = albedo_DF[albedo_DF['class'] == 3]
    CC_DF = CC_DF[CC_DF['albedo'] > CC_DF['albedo'].quantile(0.05)]
    CC_DF = CC_DF[CC_DF['albedo'] < CC_DF['albedo'].quantile(0.95)]

    WAT_DF = albedo_DF[albedo_DF['class'] == 2]
    WAT_DF = WAT_DF[WAT_DF['albedo'] > WAT_DF['albedo'].quantile(0.05)]
    WAT_DF = WAT_DF[WAT_DF['albedo'] < WAT_DF['albedo'].quantile(0.95)]

    SN_DF = albedo_DF[albedo_DF['class'] == 1]
    SN_DF = SN_DF[SN_DF['albedo'] > SN_DF['albedo'].quantile(0.05)]
    SN_DF = SN_DF[SN_DF['albedo'] > SN_DF['albedo'].quantile(0.95)]

    # Calculate summary stats
    mean_CC = CC_DF['albedo'].mean()
    std_CC = CC_DF['albedo'].std()
    max_CC = CC_DF['albedo'].max()
    min_CC = CC_DF['albedo'].min()

    mean_CI = CI_DF['albedo'].mean()
    std_CI = CI_DF['albedo'].std()
    max_CI = CI_DF['albedo'].max()
    min_CI = CI_DF['albedo'].min()

    mean_LA = LA_DF['albedo'].mean()
    std_LA = LA_DF['albedo'].std()
    max_LA = LA_DF['albedo'].max()
    min_LA = LA_DF['albedo'].min()

    mean_HA = HA_DF['albedo'].mean()
    std_HA = HA_DF['albedo'].std()
    max_HA = HA_DF['albedo'].max()
    min_HA = HA_DF['albedo'].min()

    mean_WAT = WAT_DF['albedo'].mean()
    std_WAT = WAT_DF['albedo'].std()
    max_WAT = WAT_DF['albedo'].max()
    min_WAT = WAT_DF['albedo'].min()

    mean_SN = SN_DF['albedo'].mean()
    std_SN = SN_DF['albedo'].std()
    max_SN = SN_DF['albedo'].max()
    min_SN = SN_DF['albedo'].min()

    ## FIND IDX WHERE CLASS = Hbio..
    ## BIN ALBEDOS FROM SAME IDXs
    print('mean albedo WAT = ', mean_WAT)
    print('mean albedo CC = ', mean_CC)
    print('mean albedo CI = ', mean_CI)
    print('mean albedo LA = ', mean_LA)
    print('mean albedo HA = ', mean_HA)
    print('mean albedo SN = ', mean_SN)
    print('n HA = ', len(HA_DF))
    print('n LA = ', len(LA_DF))
    print('n CI = ', len(CI_DF))
    print('n CC = ', len(CC_DF))
    print('n WAT = ', len(WAT_DF))
    print('n SN = ', len(SN_DF))

    return alb_WAT, alb_CC, alb_CI, alb_LA, alb_HA, alb_SN, mean_CC, std_CC, max_CC, min_CC, mean_CI, std_CI, max_CI,\
           min_CI, mean_LA, min_LA, max_LA, std_LA, mean_HA, std_HA, max_HA, min_HA, mean_WAT, std_WAT, max_WAT, \
           min_WAT, mean_SN, std_SN, max_SN, min_SN


################################################################################
################################################################################


############### RUN ENTIRE SEQUENCE ###################

# create dataset
X, XX, YY = create_dataset(HCRF_file, plot_spectra=False, savefigs=True)

# optimise and train model
clf = optimise_train_model(X, XX, YY, test_size=0.3)

# apply model to UAV image
predicted, albedo_array, HA_coverage, LA_coverage, CI_coverage, CC_coverage, WAT_coverage, SN_coverage = \
ImageAnalysis(img_name,clf,plot_maps = True, savefigs=False)

# obtain albedo summary stats
alb_WAT, alb_CC, alb_CI, alb_LA, alb_HA, alb_SN, mean_CC,std_CC,max_CC,min_CC,mean_CI,std_CI,max_CI,min_CI, \
mean_LA,min_LA,max_LA,std_LA,mean_HA,std_HA,max_HA,min_HA,mean_WAT,std_WAT,max_WAT,min_WAT,mean_SN,std_SN,max_SN,\
min_SN = albedo_report(predicted,albedo_array)

