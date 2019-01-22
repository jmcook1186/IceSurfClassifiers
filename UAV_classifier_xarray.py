#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:32:24 2018
@author: joseph cook


# Implementation of UAV multispectral image classification algorithm using xarray.
# Code from github.com/ajtedstone/IceSurfClassifiers was adapted and incorporated in this script.


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
# into a format appropriate for supervised classification. The raw hyperspectral data is
# first organised into separate pandas dataframes for each surface class.
# The data is then reduced down to the reflectance at the five key wavelengths
# and the remaining data discarded. The dataset is then arranged into columns
# with one column per wavelength and a separate column for the surface class.
# The dataset's features are the reflectance at each wavelength, and the labels
# are the surface types. The dataframes for each surface type are concatenated into
# one large dataframe and then the labels are removed and saved as a separate
# dataframe. No scaling of the data is required because the reflectance is already
# normalised between 0 and 1 by the spectrometer.

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



# Note:
# UAV images should be provided as netcdf files and opened using xarray
# To convert UAV tif to netcdf: gdal_translate -of netcdf uav_21_7_5cm_commongrid.tif uav_data.nc
# The UAV image has been preprocessed in Agisoft Photoscan, including stitching
# and conversion of raw DN to reflectance using calibrated reflectance panels
# on the ground.

# Note:
# Choosing to interactively plot all figures can lead to memory overload. Better to save the figures to file until this
# is fixed.

"""



import pandas as pd
import xarray as xr
import sklearn_xarray
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
from sklearn.externals import joblib
from datetime import datetime
import matplotlib as mpl
import georaster
from osgeo import gdal, osr
import seaborn as sn

# matplotlib settings: use ggplot style and turn interactive mode off so that plots can be saved and not shown (for
# rapidly processing multiple images later)

mpl.style.use('ggplot')
plt.ioff()

# Set paths to training data and image files
# LOCAL
HCRF_file = '/home/joe/Code/IceSurfClassifiers/Training_Data/HCRF_master_machine_snicar.csv'
img_file = '/home/joe/Desktop/uav_data.nc'
savefig_path = '/home/joe/Desktop/'

# VIRTUAL MACHINE
# img_file = '/home/tothepoles/PycharmProjects/IceSurfClassifiers/uav_data.nc'
# HCRF_file = '/home/tothepoles/PycharmProjects/IceSurfClassifiers/Training_Data/HCRF_master_machine_snicar.csv'
# savefig_path = '/data/home/tothepoles/Desktop/'

# set coordinates for generating training data from images

x_min = [4810, 5120, 5185, 4036, 5050]

x_max = [4850, 5160, 5200, 4052, 5075]

y_min = [1070, 750, 810, 380, 1670]

y_max = [1115, 790, 832, 415, 1720]

area_labels = [1, 1, 1, 1, 1]

n_areas = len(x_min)


# Define functions
####################

def training_data_from_spectra(HCRF_file, plot_spectra=True, savefigs=True):
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
               'HA_25', 'HA_26', 'HA_27', 'HA_28', 'HA_29', 'HA_30', 'HA_31', '13_7_S2', '14_7_SB9',
               'MA_11', 'MA_14', 'MA_15', 'MA_17', '21_7_SB2', '22_7_SB1', 'MA_4', 'MA_7', 'MA_18',
               '27_7_16_SITE3_WMELON1', '27_7_16_SITE3_WMELON3', '27_7_16_SITE2_ALG1',
               '27_7_16_SITE2_ALG2', '27_7_16_SITE2_ALG3', '27_7_16_SITE2_ICE3', '27_7_16_SITE2_ICE5',
               '27_7_16_SITE3_ALG4', '5_8_16_site2_ice7', '5_8_16_site3_ice2', '5_8_16_site3_ice3',
               '5_8_16_site3_ice5', '5_8_16_site3_ice6', '5_8_16_site3_ice7',
               '5_8_16_site3_ice8', '5_8_16_site3_ice9']


    LAsites = [
        '14_7_S2', '14_7_S3', '14_7_SB2', '14_7_SB3', '14_7_SB7', '15_7_S2', '15_7_SB4',
        '20_7_SB1', '20_7_SB3', '21_7_S1', '21_7_S5', '21_7_SB4', '22_7_SB2', '22_7_SB3', '22_7_S1',
        '23_7_S1', '23_7_S2', '24_7_S2', 'MA_1', 'MA_2', 'MA_3', 'MA_5', 'MA_6', 'MA_8', 'MA_9',
        'MA_10', 'MA_12', 'MA_13', 'MA_16', 'MA_19', '13_7_S1', '13_7_S3', '14_7_S1', '15_7_S1',
        '15_7_SB2', '20_7_SB2', '21_7_SB5', '21_7_SB8', '25_7_S3', '5_8_16_site2_ice10', '5_8_16_site2_ice5',
        '5_8_16_site2_ice9', '27_7_16_SITE3_WHITE3']

    CIsites = ['21_7_S4', '13_7_SB3', '15_7_S4', '15_7_SB1', '15_7_SB5', '21_7_S2',
               '21_7_SB3', '22_7_S2', '22_7_S4', '23_7_SB1', '23_7_SB2', '23_7_S4',
               'WI_1', 'WI_2', 'WI_4', 'WI_5', 'WI_6', 'WI_7', 'WI_9', 'WI_10', 'WI_11',
               'WI_12', 'WI_13', '27_7_16_SITE3_WHITE1', '27_7_16_SITE3_WHITE2',
               '27_7_16_SITE2_ICE2', '27_7_16_SITE2_ICE4', '27_7_16_SITE2_ICE6',
               '5_8_16_site2_ice1', '5_8_16_site2_ice2', '5_8_16_site2_ice3', '5_8_16_site2_ice4',
               '5_8_16_site2_ice6', '5_8_16_site2_ice8', '5_8_16_site3_ice1', '5_8_16_site3_ice4']

    CCsites = ['DISP1', 'DISP2', 'DISP3', 'DISP4', 'DISP5', 'DISP6', 'DISP7', 'DISP8',
               'DISP9', 'DISP10', 'DISP11', 'DISP12', 'DISP13', 'DISP14', '27_7_16_SITE3_DISP1', '27_7_16_SITE3_DISP3',]

    WATsites = ['21_7_SB5', '21_7_SB8', 'WAT_1', 'WAT_3', 'WAT_6']

    SNsites = ['14_7_S4', '14_7_SB6', '14_7_SB8', '17_7_SB2',
               '27_7_16_KANU_', '27_7_16_SITE2_1', '5_8_16_site1_snow10', '5_8_16_site1_snow2', '5_8_16_site1_snow3',
               '5_8_16_site1_snow4', '5_8_16_site1_snow6',
               '5_8_16_site1_snow7', '5_8_16_site1_snow9', 'SNICAR100', 'SNICAR200',
               'SNICAR300', 'SNICAR400', 'SNICAR500', 'SNICAR600', 'SNICAR700', 'SNICAR800', 'SNICAR900', 'SNICAR1000',]


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

    if plot_spectra or savefigs:

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
            plt.savefig(str(savefig_path + "training_spectra.png"))

        if plot_spectra:
            plt.show()


    X = pd.DataFrame()

    X['Band1'] = np.array(HA_hcrf.iloc[125])
    X['Band2'] = np.array(HA_hcrf.iloc[210])
    X['Band3'] = np.array(HA_hcrf.iloc[318])
    X['Band4'] = np.array(HA_hcrf.iloc[367])
    X['Band5'] = np.array(HA_hcrf.iloc[490])

    X['label'] = 6

    Y = pd.DataFrame()
    Y['Band1'] = np.array(LA_hcrf.iloc[125])
    Y['Band2'] = np.array(LA_hcrf.iloc[210])
    Y['Band3'] = np.array(LA_hcrf.iloc[318])
    Y['Band4'] = np.array(LA_hcrf.iloc[367])
    Y['Band5'] = np.array(LA_hcrf.iloc[490])

    Y['label'] = 5

    Z = pd.DataFrame()

    Z['Band1'] = np.array(CI_hcrf.iloc[125])
    Z['Band2'] = np.array(CI_hcrf.iloc[210])
    Z['Band3'] = np.array(CI_hcrf.iloc[318])
    Z['Band4'] = np.array(CI_hcrf.iloc[367])
    Z['Band5'] = np.array(CI_hcrf.iloc[490])

    Z['label'] = 4

    P = pd.DataFrame()

    P['Band1'] = np.array(CC_hcrf.iloc[125])
    P['Band2'] = np.array(CC_hcrf.iloc[210])
    P['Band3'] = np.array(CC_hcrf.iloc[318])
    P['Band4'] = np.array(CC_hcrf.iloc[367])
    P['Band5'] = np.array(CC_hcrf.iloc[490])

    P['label'] = 3

    Q = pd.DataFrame()
    Q['Band1'] = np.array(WAT_hcrf.iloc[125])
    Q['Band2'] = np.array(WAT_hcrf.iloc[210])
    Q['Band3'] = np.array(WAT_hcrf.iloc[318])
    Q['Band4'] = np.array(WAT_hcrf.iloc[367])
    Q['Band5'] = np.array(WAT_hcrf.iloc[490])

    Q['label'] = 2

    R = pd.DataFrame()
    R['Band1'] = np.array(SN_hcrf.iloc[125])
    R['Band2'] = np.array(SN_hcrf.iloc[210])
    R['Band3'] = np.array(SN_hcrf.iloc[318])
    R['Band4'] = np.array(SN_hcrf.iloc[367])
    R['Band5'] = np.array(SN_hcrf.iloc[490])

    R['label'] = 1

    Zero = pd.DataFrame()
    Zero['Band1'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['Band2'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['Band3'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['Band4'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['Band5'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    Zero['label'] = 0

    # Join dataframes into one continuous DF
    X = X.append(Y, ignore_index=True)
    X = X.append(Z, ignore_index=True)
    X = X.append(P, ignore_index=True)
    X = X.append(Q, ignore_index=True)
    X = X.append(R, ignore_index=True)
    X = X.append(Zero, ignore_index=True)

    return X


def training_data_from_img(X, img_file, x_min, x_max, y_min, y_max, n_areas, area_labels):

    """Function appends to training data spectra from selected homogenous areas from images
     where the surface label is known. Currently only appending to the SN class because of availability of unambiguous
     sites """


    with xr.open_dataset(img_file) as uav:
        for i in range(n_areas):
            npix = (x_max[i]- x_min[i])*(y_max[i]- y_min[i])

            # slice areas defined by corner coordinates
            uavsubsetB1 = uav.Band1[x_min[i]:x_max[i], y_min[i]: y_max[i]]
            uavsubsetB2 = uav.Band2[x_min[i]:x_max[i], y_min[i]: y_max[i]]
            uavsubsetB3 = uav.Band3[x_min[i]:x_max[i], y_min[i]: y_max[i]]
            uavsubsetB4 = uav.Band4[x_min[i]:x_max[i], y_min[i]: y_max[i]]
            uavsubsetB5 = uav.Band5[x_min[i]:x_max[i], y_min[i]: y_max[i]]

            stackB1 = uavsubsetB1.stack(z=('x','y'))
            stackB2 = uavsubsetB2.stack(z=('x','y'))
            stackB3 = uavsubsetB3.stack(z=('x','y'))
            stackB4 = uavsubsetB4.stack(z=('x','y'))
            stackB5 = uavsubsetB5.stack(z=('x','y'))

            label = area_labels[i]

            tempDF = pd.DataFrame()
            tempDF['Band1'] = stackB1.values
            tempDF['Band2'] = stackB2.values
            tempDF['Band3'] = stackB3.values
            tempDF['Band4'] = stackB4.values
            tempDF['Band5'] = stackB5.values
            tempDF['label'] = label

            X = X.append(tempDF,ignore_index=False)

    return X, tempDF

def split_train_test(X, test_size=0.2, n_trees= 64, print_conf_mx = True, plot_conf_mx = True, savefigs = False,
                     show_model_performance = True, pickle_model=False):
    """ Split spectra into training and testing data sets
    Arguments:
    spectra : pd.DataFrame of spectra (each spectra = row, columns = bands)
    test_size
    returns training and testing datasets
    """

    # Split into test and train datasets
    features = X.drop(labels=['label'], axis=1)
    labels = X.filter(items=['label'])
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features, labels,
        test_size=test_size)

    # Convert training and test datasets to DataArrays
    X_train_xr = xr.DataArray(X_train, dims=('samples','bands'), coords={'bands':features.columns})
    Y_train_xr = xr.DataArray(Y_train, dims=('samples','label'))
    X_test_xr = xr.DataArray(X_test, dims=('samples','bands'), coords={'bands':features.columns})
    Y_test_xr = xr.DataArray(Y_test, dims=('samples','label'))


    # Define classifier
    clf = sklearn_xarray.wrap(
        RandomForestClassifier(n_estimators=n_trees, max_leaf_nodes=None, n_jobs=-1),
        sample_dim='samples', reshapes='bands')

    # fot classifier to training data
    clf.fit(X_train_xr, Y_train_xr)

    # test model performance
    accuracy_RF = clf.score(X_train_xr, Y_train_xr)
    Y_predict_RF = clf.predict(X_train_xr)
    conf_mx_RF = confusion_matrix(Y_train_xr, Y_predict_RF)
    recall_RF = recall_score(Y_train_xr, Y_predict_RF, average="weighted")
    f1_RF = f1_score(Y_train_xr, Y_predict_RF, average="weighted")
    precision_RF = precision_score(Y_train_xr, Y_predict_RF, average='weighted')
    average_metric_RF = (accuracy_RF + recall_RF + f1_RF) / 3

    if show_model_performance:
        print("\nModel Performance", "\n", "\nRandom Forest accuracy = ", accuracy_RF, "\nRandom Forest F1 Score = ", f1_RF,
              "\nRandom Forest Recall = ",
              recall_RF, "\nRandom Forest Precision = ", precision_RF, "\naverage of all metrics = ", average_metric_RF)

    # calculate normalised confusion matrix
    row_sums = conf_mx_RF.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx_RF / row_sums
    np.fill_diagonal(norm_conf_mx, 0)

    # plot confusion matrices as subplots in a single figure using Seaborn heatmap
    if plot_conf_mx or savefigs:

        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(35,35))
        sn.heatmap(conf_mx_RF, annot=True, annot_kws={"size": 16},
                   xticklabels=['Unknown', 'Snow', 'Water', 'Cryoconite','Clean Ice', 'Light Algae', 'Heavy Algae'],
                   yticklabels=['Unknown', 'Snow', 'Water', 'Cryoconite', 'Clean Ice', 'Light Algae', 'Heavy Algae'],
                   cbar_kws={"shrink": 0.4, 'label':'frequency'}, ax=ax1), ax1.tick_params(axis='both', rotation=45)
        ax1.set_title('Confusion Matrix'), ax1.set_aspect('equal')

        sn.heatmap(norm_conf_mx, annot=True, annot_kws={"size": 16}, cmap=plt.cm.gray,
                   xticklabels=['Unknown', 'Snow', 'Water', 'Cryoconite', 'Clean Ice', 'Light Algae', 'Heavy Algae'],
                   yticklabels=['Unknown', 'Snow', 'Water', 'Cryoconite', 'Clean Ice', 'Light Algae', 'Heavy Algae'],
                   cbar_kws={"shrink": 0.4, 'label':'Normalised Error'}, ax=ax2), ax2.tick_params(axis='both', rotation=45)
        ax2.set_title('Normalised Confusion Matrix'), ax2.set_aspect('equal')
        plt.tight_layout()

        if savefigs:
            plt.savefig(str(savefig_path + "final_model_confusion_matrices.png"))

        if plot_conf_mx:
            plt.show()


    if print_conf_mx:
        print('Final Confusion Matrix')
        print(conf_mx_RF)
        print()
        print('Normalised Confusion Matrix')
        print(norm_conf_mx)


    if pickle_model:
        # pickle the classifier model for archiving or for reusing in another code
        joblibfile = 'Sentinel2_classifier.pkl'
        joblib.dump(clf, joblibfile)

        # to load this classifier into another code use the following syntax:
        # clf = joblib.load(joblib_file)

    return clf, conf_mx_RF, norm_conf_mx



def classify_images(clf, img_file, plot_maps = True, savefigs = False, save_netcdf = False):
    startTime = datetime.now()  # start timer

    # open uav file using xarray. Use "with ... as ..." method so that file auto-closes after use
    with xr.open_dataset(img_file) as uav:

        # calibration against ASD Field Spec
        uav['Band1'] -= 0.17
        uav['Band2'] -= 0.18
        uav['Band3'] -= 0.15
        uav['Band4'] -= 0.16
        uav['Band5'] -= 0.05

        # Set index for reducing data
        band_idx = pd.Index([1, 2, 3, 4, 5], name='bands')

        # concatenate the bands into a single dimension ('bands_idx') in the data array
        concat = xr.concat([uav.Band1, uav.Band2, uav.Band3, uav.Band4, uav.Band5], band_idx)

        # Mask nodata areas
        concat2 = concat.where(concat.sum(dim='bands') > 0)

        # stack the values into a 1D array
        stacked = concat2.stack(allpoints=['y', 'x'])

        # Transpose and rename so that DataArray has exactly the same layout/labels as the training DataArray.
        stackedT = stacked.T
        stackedT = stackedT.rename({'allpoints': 'samples'})
        stackedT = stackedT.where(stackedT.sum(dim='bands') > 0).dropna(dim='samples')

        # apply classifier (make use of all cores)
        predicted_temp = clf.predict(stackedT)
        # Unstack back to x,y grid and save as numpy array
        predicted = np.array(predicted_temp.unstack(dim='samples'))

        # convert albedo array to numpy array for analysis
        # obtain albedo by aplying Knap (1999) narrowband to broadband albedo conversion.
        concat = np.array(concat)
        albedo = 0.726 * (concat[1,:,:] - 0.18) - 0.322 * (
                concat[1,:,:] - 0.18) ** 2 - 0.015 * (concat[3,:,:] - 0.16) + 0.581 \
                   * (concat[3,:,:] - 0.16)

        # convert albedo array to numpy array for analysis
        albedo[albedo < -0.48] = None  # areas outside of main image area identified set to null
        with np.errstate(divide='ignore', invalid='ignore'):  # ignore warning about nans in array
            albedo[albedo < 0] = 0  # set any subzero pixels inside image area to 0



    # collate predicted map, albedo map and projection info into xarray dataset
    # 1) Retrieve projection info from uav datafile and add to netcdf
    srs = osr.SpatialReference()
    srs.ImportFromProj4('+init=epsg:32623')
    proj_info = xr.DataArray(0, encoding={'dtype': np.dtype('int8')})
    proj_info.attrs['projected_crs_name'] = srs.GetAttrValue('projcs')
    proj_info.attrs['grid_mapping_name'] = 'UTM'
    proj_info.attrs['scale_factor_at_central_origin'] = srs.GetProjParm('scale_factor')
    proj_info.attrs['standard_parallel'] = srs.GetProjParm('latitude_of_origin')
    proj_info.attrs['straight_vertical_longitude_from_pole'] = srs.GetProjParm('central_meridian')
    proj_info.attrs['false_easting'] = srs.GetProjParm('false_easting')
    proj_info.attrs['false_northing'] = srs.GetProjParm('false_northing')
    proj_info.attrs['latitude_of_projection_origin'] = srs.GetProjParm('latitude_of_origin')

    # 2) Create associated lat/lon coordinates DataArrays usig georaster (imports geo metadata without loading img)
    # see georaster docs at https: // media.readthedocs.org / pdf / georaster / latest / georaster.pdf
    uav = georaster.SingleBandRaster('NETCDF:"%s":Band1' % (img_file),
                                     load_data=False)
    lon, lat = uav.coordinates(latlon=True)
    uav = None # close file
    uav = xr.open_dataset(img_file, chunks={'x': 2000, 'y': 2000})
    coords_geo = {'y': uav['y'], 'x': uav['x']}
    uav = None #close file

    lon_array = xr.DataArray(lon, coords=coords_geo, dims=['y', 'x'],
                          encoding={'_FillValue': -9999., 'dtype': 'int16', 'scale_factor': 0.000000001})
    lon_array.attrs['grid_mapping'] = 'UTM'
    lon_array.attrs['units'] = 'degrees'
    lon_array.attrs['standard_name'] = 'longitude'

    lat_array = xr.DataArray(lat, coords=coords_geo, dims=['y', 'x'],
                          encoding={'_FillValue': -9999., 'dtype': 'int16', 'scale_factor': 0.000000001})
    lat_array.attrs['grid_mapping'] = 'UTM'
    lat_array.attrs['units'] = 'degrees'
    lat_array.attrs['standard_name'] = 'latitude'

    # 3) add predicted map array and add metadata
    predictedxr = xr.DataArray(predicted, coords=coords_geo, dims=['y','x'])
    predictedxr.encoding = {'dtype': 'int16', 'zlib': True, '_FillValue': -9999}
    predictedxr.name = 'Surface Class'
    predictedxr.attrs['long_name'] = 'Surface classified using Random Forest'
    predictedxr.attrs['units'] = 'None'
    predictedxr.attrs[
        'key'] = 'Unknown:0; Snow:1; Water:2; Cryoconite:3; Clean Ice:4; Light Algae:5; Heavy Algae:6'
    predictedxr.attrs['grid_mapping'] = 'UTM'

    # add albedo map array and add metadata
    albedoxr = xr.DataArray(albedo, coords=coords_geo, dims=['y', 'x'])
    albedoxr.encoding = {'dtype': 'int16', 'scale_factor': 0.01, 'zlib': True, '_FillValue': -9999}
    albedoxr.name = 'Surface albedo computed after Knap et al. (1999) narrowband-to-broadband conversion'
    albedoxr.attrs['units'] = 'dimensionless'
    albedoxr.attrs['grid_mapping'] = 'UTM'

    # collate data arrays into a dataset
    dataset = xr.Dataset({

       'classified': (['x', 'y'],predictedxr),
        'albedo': (['x', 'y'], albedoxr),
        'Projection(UTM)': proj_info,
        'longitude': (['x','y'],lon_array),
        'latitude': (['x','y'],lat_array)
    })

    # add metadata for dataset
    dataset.attrs['Conventions'] = 'CF-1.4'
    dataset.attrs['Author'] = 'Joseph Cook (University of Sheffield, UK)'
    dataset.attrs[
        'title'] = 'Classified surface and albedo maps produced from UAV-derived multispectral ' \
                   'imagery of the SW Greenland Ice Sheet'

    # Additional geo-referencing
    dataset.attrs['nx'] = len(dataset.x)
    dataset.attrs['ny'] = len(dataset.y)
    dataset.attrs['xmin'] = float(dataset.x.min())
    dataset.attrs['ymax'] = float(dataset.y.max())
    dataset.attrs['spacing'] = 0.05

    # NC conventions metadata for dimensions variables
    dataset.x.attrs['units'] = 'meters'
    dataset.x.attrs['standard_name'] = 'projection_x_coordinate'
    dataset.x.attrs['point_spacing'] = 'even'
    dataset.x.attrs['axis'] = 'x'

    dataset.y.attrs['units'] = 'meters'
    dataset.y.attrs['standard_name'] = 'projection_y_coordinate'
    dataset.y.attrs['point_spacing'] = 'even'
    dataset.y.attrs['axis'] = 'y'
    print('time taken to create dataset: ',datetime.now()-startTime)

    # save dataset to netcdf if requested
    if save_netcdf:
        dataset.to_netcdf(savefig_path + "Classification_and_Albedo_Data.nc")

    # plot and save figure if requested
    if plot_maps or savefigs:
        # set color scheme for plots - custom for predicted
        cmap1 = mpl.colors.ListedColormap(
            ['purple', 'white', 'royalblue', 'black', 'lightskyblue', 'mediumseagreen', 'darkgreen'])
        cmap1.set_under(color='white')  # make sure background is white
        cmap2 = plt.get_cmap('Greys_r')  # reverse greyscale for albedo
        cmap2.set_under(color='white')  # make sure background is white

        fig = plt.figure(figsize=(25, 25))
        plt.title("Classified ice surface and its albedos from UAV imagery: SW Greenland Ice Sheet", fontsize=28)
        class_labels = ['Unknown', 'Snow', 'Water', 'Cryoconite', 'Clean Ice', 'Light Algae', 'Heavy Algae']

        # first subplot = classified map
        ax1 = plt.subplot(211)
        img = dataset.classified.plot(cmap=cmap1, add_colorbar=False)
        cbar = fig.colorbar(mappable=img, ax=ax1)
        # workaround to get colorbar labels centrally positioned
        n_classes = 7
        tick_locs = (np.arange(n_classes) + 0.5) * (n_classes - 1) / n_classes
        cbar.set_ticks(tick_locs)
        cbar.ax.set_yticklabels(class_labels, rotation=45, va='center')
        plt.title('Classified Surface Map (UTM coordinates)'), ax1.set_aspect('equal')

        # second subplot = albedo map
        ax2 = plt.subplot(212)
        dataset.albedo.plot(cmap=cmap2, vmin=0, vmax=1, ax=ax2), plt.title('Albedo Map (UTM coordinates)')
        ax2.set_aspect('equal')

        if savefigs:
            plt.savefig(str(savefig_path + "UAV_classified_albedo_map.png"), dpi=150)

        if plot_maps:
            plt.show()

    print("\n Image Classification and Albedo Function Time = ", datetime.now() - startTime)

    return predicted, albedo


def albedo_report(predicted, albedo, save_albedo_data = False):

    # match albedo to predicted class using indexes
    predicted = np.array(predicted).ravel()
    albedo = np.array(albedo).ravel()

    albedoDF = pd.DataFrame()
    albedoDF['pred'] = predicted
    albedoDF['albedo'] = albedo
    albedoDF = albedoDF.dropna()

    #coverage statistics
    HApercent = (albedoDF['pred'][albedoDF['pred']==6].count()) / (albedoDF['pred'][albedoDF['pred']!=0].count())
    LApercent = (albedoDF['pred'][albedoDF['pred']==5].count()) / (albedoDF['pred'][albedoDF['pred']!=0].count())
    CIpercent = (albedoDF['pred'][albedoDF['pred']==4].count()) / (albedoDF['pred'][albedoDF['pred']!=0].count())
    CCpercent = (albedoDF['pred'][albedoDF['pred']==3].count()) / (albedoDF['pred'][albedoDF['pred']!=0].count())
    WATpercent = (albedoDF['pred'][albedoDF['pred'] == 2].count()) / (albedoDF['pred'][albedoDF['pred'] != 0].count())
    SNpercent = (albedoDF['pred'][albedoDF['pred']==1].count()) / (albedoDF['pred'][albedoDF['pred']!=0].count())


    if save_albedo_data:
        albedoDF.to_csv(savefig_path + 'RawAlbedoData.csv')
        albedoDF.groupby(['pred']).count().to_csv(savefig_path+'Surface_Type_Counts.csv')
        albedoDF.groupby(['pred']).describe()['albedo'].to_csv(savefig_path+'Albedo_summary_stats.csv')

    # report summary stats
    print('\n Surface type counts: \n', albedoDF.groupby(['pred']).count())
    print('\n Summary Statistics for ALBEDO of each surface type: \n',albedoDF.groupby(['pred']).describe()['albedo'])

    print('\n "Percent coverage by surface type: \n')
    print(' HA coverage = ',np.round(HApercent,2)*100,'%\n','LA coverage = ',np.round(LApercent,2)*100,'%\n','CI coverage = ',
          np.round(CIpercent,2)*100,'%\n', 'CC coverage = ',np.round(CCpercent,2)*100,'%\n', 'SN coverage = ',
          np.round(SNpercent,2)*100,'%\n', 'WAT coverage = ', np.round(WATpercent,2)*100,'%\n', 'Total Algal Coverage = ',
          np.round(HApercent+LApercent,2)*100)

    return albedoDF



X = training_data_from_spectra(HCRF_file, plot_spectra=False, savefigs=False)

X, tempDF = training_data_from_img(X = X, img_file = img_file, x_min = x_min, x_max = x_max, y_min = y_min,
                                  y_max = y_max, area_labels = area_labels, n_areas=n_areas)


clf, conf_mx_RF, norm_conf_mx = split_train_test(X, test_size=0.2, n_trees=32, print_conf_mx = False, plot_conf_mx = False,
                                                   savefigs = False, show_model_performance = True, pickle_model=False)

#predicted, albedo = classify_images(clf, img_file, plot_maps = False, savefigs = True, save_netcdf = False)

# albedoDF = albedo_report(predicted, albedo, save_albedo_data = False)