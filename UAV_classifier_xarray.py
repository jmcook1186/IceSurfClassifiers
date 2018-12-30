#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:32:24 2018

@author: joseph cook
"""

import pandas as pd
import xarray as xr
import sklearn_xarray
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, osr
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import matplotlib as mpl
mpl.style.use('ggplot')

# Note:
# To convert UAV tif to netcdf: gdal_translate -of netcdf uav_21_7_5cm_commongrid.tif uav_data.nc

plt.ioff() # turn interactive plotting off so that plots can be saved but not shown

#Paths

# local paths
HCRF_file = '/home/joe/Code/IceSurfClassifiers/Training_Data/HCRF_master_machine_snicar.csv'
img_name = '/home/joe/Desktop/uav_data.nc'
savefig_path = '//home/joe/Desktop/'

# VM paths
# img_file = '/home/tothepoles/PycharmProjects/IceSurfClassifiers/uav_data.nc'
# HCRF_file = '/home/tothepoles/PycharmProjects/IceSurfClassifiers/Training_Data/HCRF_master_machine_snicar.csv'
# savefig_path = '/data/home/tothepoles/Desktop/'

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
            plt.savefig(str(savefig_path + "training_spectra.jpg"))

        if plot_spectra:
            plt.show()

    X = pd.DataFrame()

    X['Band1'] = np.array(HA_hcrf.iloc[125])
    X['Band2'] = np.array(HA_hcrf.iloc[210])
    X['Band3'] = np.array(HA_hcrf.iloc[318])
    X['Band4'] = np.array(HA_hcrf.iloc[367])
    X['Band5'] = np.array(HA_hcrf.iloc[490])

    X['label'] = 'HA'

    Y = pd.DataFrame()
    Y['Band1'] = np.array(LA_hcrf.iloc[125])
    Y['Band2'] = np.array(LA_hcrf.iloc[210])
    Y['Band3'] = np.array(LA_hcrf.iloc[318])
    Y['Band4'] = np.array(LA_hcrf.iloc[367])
    Y['Band5'] = np.array(LA_hcrf.iloc[490])

    Y['label'] = 'LA'

    Z = pd.DataFrame()

    Z['Band1'] = np.array(CI_hcrf.iloc[125])
    Z['Band2'] = np.array(CI_hcrf.iloc[210])
    Z['Band3'] = np.array(CI_hcrf.iloc[318])
    Z['Band4'] = np.array(CI_hcrf.iloc[367])
    Z['Band5'] = np.array(CI_hcrf.iloc[490])

    Z['label'] = 'CI'

    P = pd.DataFrame()

    P['Band1'] = np.array(CC_hcrf.iloc[125])
    P['Band2'] = np.array(CC_hcrf.iloc[210])
    P['Band3'] = np.array(CC_hcrf.iloc[318])
    P['Band4'] = np.array(CC_hcrf.iloc[367])
    P['Band5'] = np.array(CC_hcrf.iloc[490])

    P['label'] = 'CC'

    Q = pd.DataFrame()
    Q['Band1'] = np.array(WAT_hcrf.iloc[125])
    Q['Band2'] = np.array(WAT_hcrf.iloc[210])
    Q['Band3'] = np.array(WAT_hcrf.iloc[318])
    Q['Band4'] = np.array(WAT_hcrf.iloc[367])
    Q['Band5'] = np.array(WAT_hcrf.iloc[490])

    Q['label'] = 'WAT'

    R = pd.DataFrame()
    R['Band1'] = np.array(SN_hcrf.iloc[125])
    R['Band2'] = np.array(SN_hcrf.iloc[210])
    R['Band3'] = np.array(SN_hcrf.iloc[318])
    R['Band4'] = np.array(SN_hcrf.iloc[367])
    R['Band5'] = np.array(SN_hcrf.iloc[490])

    R['label'] = 'SN'

    Zero = pd.DataFrame()
    Zero['Band1'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['Band2'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['Band3'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['Band4'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['Band5'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    Zero['label'] = 'UNKNOWN'

    # Join dataframes into one continuous DF
    X = X.append(Y, ignore_index=True)
    X = X.append(Z, ignore_index=True)
    X = X.append(P, ignore_index=True)
    X = X.append(Q, ignore_index=True)
    X = X.append(R, ignore_index=True)
    X = X.append(Zero, ignore_index=True)

    return X




def train_test_split(X, test_size=0.2):
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
    X_train_xr = xr.DataArray(X_train, dims=('samples','b'), coords={'b':features.columns})
    Y_train_xr = xr.DataArray(Y_train, dims=('samples','label'))
    X_test_xr = xr.DataArray(X_test, dims=('samples','b'), coords={'b':features.columns})
    Y_test_xr = xr.DataArray(Y_test, dims=('samples','label'))

    return X_train_xr, Y_train_xr, X_test_xr, Y_test_xr



def train_RF(X_train_xr, Y_train_xr):
    """
    Train a Random Forest Classifier (wrapped for xarray)
    """
    clf = sklearn_xarray.wrap(
        RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1),
        sample_dim='samples', reshapes='b')

    clf.fit(X_train_xr, Y_train_xr)

    return clf



def classify_images(clf, img_file, plot_maps = True, savefigs = False, save_netcdf = False):

    startTime = datetime.now()

    uav = xr.open_dataset(img_file, chunks={'x': 2000, 'y': 2000})

    # # correct reflectance according to calibration against ASD Field Spec

    uav['Band1'] -= 0.17
    uav['Band2'] -= 0.18
    uav['Band3'] -= 0.15
    uav['Band4'] -= 0.16
    uav['Band5'] -= 0.1

    # Set index for reducing data

    b_ix = pd.Index([1, 2, 3, 4, 5], name='b')

    # concatenate the bands into a single dimension in the data array
    concat = xr.concat([uav.Band1, uav.Band2, uav.Band3, uav.Band4, uav.Band5], b_ix)

    # Mask nodata areas
    concat = concat.where(concat.sum(dim='b') > 0)

    # stack the values into a 1D array
    stacked = concat.stack(allpoints=['y', 'x'])

    # Transpose and rename so that DataArray has exactly the same layout/labels as the training DataArray.
    stackedT = stacked.T
    stackedT = stackedT.rename({'allpoints': 'samples'})
    stackedT = stackedT.where(stackedT.sum(dim='b') > 0).dropna(dim='samples')

    # apply classifier
    predicted = clf.predict(stackedT).compute()

    # Unstack back to x,y grid
    predictedxr = predicted.unstack(dim='samples')

    if save_netcdf:
        predictedxr.to_netcdf(savefig_path+"classified_surface.nc")

    # calculate albedo using Knap (1999) narrowband to broadband conversion
    albedo = 0.726 * (uav['Band2'] - 0.18) - 0.322 * (uav['Band2'] - 0.18) ** 2 - 0.015 * (
                uav['Band4'] - 0.16) + 0.581 * (uav['Band4'] - 0.16)


    predicted = np.array(predictedxr)
    albedo = np.array(albedo)

    predicted[predicted == 'UNKNOWN'] = float(0)
    predicted[predicted == 'SN'] = float(1)
    predicted[predicted == 'WAT'] = float(2)
    predicted[predicted == 'CC'] = float(3)
    predicted[predicted == 'CI'] = float(4)
    predicted[predicted == 'LA'] = float(5)
    predicted[predicted == 'HA'] = float(6)

    predicted = predicted.astype(float)

    print("\nTime taken to classify image = ", datetime.now() - startTime)

    if plot_maps or savefigs:

        # set color scheme for plots - custom for predicted
        cmap1 = mpl.colors.ListedColormap(
            ['white','white', 'slategray', 'black', 'lightsteelblue', 'gold', 'orangered'])
        cmap1.set_under(color='white')  # make sure background is white
        cmap2 = plt.get_cmap('Greys_r')  # reverse greyscale for albedo
        cmap2.set_under(color='white')  # make sure background is white

        plt.figure(figsize=(25, 25))
        plt.title("Classified ice surface and its albedos from UAV imagery: SW Greenland Ice Sheet", fontsize=28)

        plt.subplot(211)
        plt.imshow(predicted, cmap=cmap1), plt.grid(None), plt.colorbar(), plt.title("UAV Classified Map")
        plt.xticks(None), plt.yticks(None)

        plt.subplot(212)
        plt.imshow(albedo, cmap=cmap2, vmin=0.000001, vmax=1), plt.grid(None), plt.colorbar(), \
        plt.xticks(None), plt.yticks(None)
        plt.title("UAV Albedo Map")

        if plot_maps:
            plt.show()

        if savefigs:
            plt.savefig(str(savefig_path + "UAV_classified_albedo_map.jpg"), dpi=300)


    print('Time taken to classify image = ', datetime.now() - startTime)

    return predictedxr, predicted, albedo




def albedo_report(predicted, albedo):

    # caluclate coverage statistics

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


    alb_WAT = []
    alb_CC = []
    alb_CI = []
    alb_LA = []
    alb_HA = []
    alb_SN = []


# albedo report

    predicted = np.array(predicted).ravel()
    albedo = np.array(albedo).ravel()

    idx_SN = np.where(predicted == 1)[0]
    idx_WAT = np.where(predicted == 2)[0]
    idx_CC = np.where(predicted == 3)[0]
    idx_CI = np.where(predicted == 4)[0]
    idx_LA = np.where(predicted == 5)[0]
    idx_HA = np.where(predicted == 6)[0]

    for i in idx_WAT:
        alb_WAT.append(albedo[i])
    for i in idx_CC:
        alb_CC.append(albedo[i])
    for i in idx_CI:
        alb_CI.append(albedo[i])
    for i in idx_LA:
        alb_LA.append(albedo[i])
    for i in idx_HA:
        alb_HA.append(albedo[i])
    for i in idx_SN:
        alb_SN.append(albedo[i])

    # create pandas dataframe containing albedo data (delete rows where albedo <= 0)
    albedo_DF = pd.DataFrame(columns=['albedo', 'class'])
    albedo_DF['class'] = predicted
    albedo_DF['albedo'] = albedo
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

    return HA_coverage, LA_coverage, CI_coverage, CC_coverage, WAT_coverage, SN_coverage, alb_WAT, alb_CC, alb_CI,\
           alb_LA, alb_HA, alb_SN, mean_CC, std_CC, max_CC, min_CC, mean_CI, std_CI, max_CI, min_CI, mean_LA, min_LA,\
           max_LA, std_LA, mean_HA, std_HA, max_HA, min_HA, mean_WAT, std_WAT, max_WAT, min_WAT, mean_SN, std_SN,\
           max_SN, min_SN



X = create_dataset(HCRF_file , plot_spectra=False, savefigs=False)

X_train_xr, Y_train_xr, X_test_xr, Y_test_xr = train_test_split(X, test_size=0.3)

clf = train_RF(X_train_xr, Y_train_xr)

predicted_unstacked, predicted, albedo,= classify_images(clf, img_file, plot_maps = False, savefigs = True, save_netcdf = True)

HA_coverage, LA_coverage, CI_coverage, CC_coverage, WAT_coverage, SN_coverage, alb_WAT, alb_CC, alb_CI,\
       alb_LA, alb_HA, alb_SN, mean_CC, std_CC, max_CC, min_CC, mean_CI, std_CI, max_CI, min_CI, mean_LA, min_LA,\
       max_LA, std_LA, mean_HA, std_HA, max_HA, min_HA, mean_WAT, std_WAT, max_WAT, min_WAT, mean_SN, std_SN,\
       max_SN, min_SN = albedo_report(predicted, albedo)


