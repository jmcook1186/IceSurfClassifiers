# The requisite datafiles are HRCF_master_machone_snicar.csv (the training data)
# and the individual Sentinel-2 band images (B02_20m.jp2, B03_20m.jp2 etc). These
# are availabe to download at github.com/jmcook1186/IceSurfClassifiers/

# This code provides functions for reading in directional reflectance data obtained
# via ground spectroscopy, reformatting into a pandas dataframe of features and labels,
# optimising and training a series of machine learning algorithms on the ground spectra
# then using the trained model to predict the surface type in each pixel of a Sentinel-2
# image. The Sentinel-2 image has been preprocessed using ESA Sen2Cor command line
# algorithm to convert to surface reflectance before being saved as a multiband TIF which
# is then loaded here. Three individual sub-areas within the main image are selected
# for analysis, maximising the glaciated area included in the study.

# This script is divided into several functions. The first preprocesses the raw data
# into a format appropriate for machine learning. The raw hyperspectral data is
# first organised into separate pandas dataframes for each surface class.
# The data is then reduced down to the reflectance at the nine key wavelengths
# and the remaining data discarded. The dataset is then arranged into columns
# with one column per wavelength and a separate column for the surface class.
# The dataset's features are the reflectance at each wavelength, and the labels
# are the surface types. The dataframes for each surface type are merged into
# one large dataframe and then the labels are removed and saved as a separate
# dataframe. XX contains all the data features, YY contains the labels only. No
# scaling of the data is required because the reflectance is already normalised
# between 0 and 1 by the spectrometer.

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

# A narrowband to broadband conversion (Knap 1999) is applied to the
# data to create an albedo map, and this is then used to create a large dataset of surface
# type and associated broadband albedo

# Parellelization

# This version of the code uses Dask to parallelize the model prediction function. Applying the trained model
# to the satellite or UAV imagery is by far the most computationally demanding function.  This breaks the
# Sentinel image into chunks and analyses them in parallel. Currently this is achived locally using the available
# cores and threads on the local machine; however, it is also formatted to interface with Kubernetes for deployment
# across available cores on MS Azure Data Science VM. Parallelizing locally (Intel® Core™ i7-7700HQ CPU @ 2.80GHz × 8,
# 32GB RAM) reduces the computation time for the entire sequence of functions from >18 minutes to <10 minutes.

# For more info http://ml.dask.org/examples/parallel-prediction.html


###########################################################################################
############################# IMPORT MODULES #########################################

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import rasterio
from datetime import datetime
from dask_ml.wrappers import ParallelPostFit

HCRF_file = '/home/joe/Code/IceSurfClassifiers/Training_Data/HCRF_master_machine_snicar.csv'
savefig_path = '//home/joe/Desktop/'


def create_dataset(HCRF_file, year=2016, plot_spectra=True, savefigs=True):
    # Two options for filepaths depending on whether the code is run for Sentinel2
    # images from 2016 or 2017

    if year == 2016:

        Sentinel_jp2s = ['/home/joe/Code/IceSurfClassifiers/2016_Sentinel/B02_20m.jp2',
                         '/home/joe/Code/IceSurfClassifiers/2016_Sentinel/B03_20m.jp2',
                         '/home/joe/Code/IceSurfClassifiers/2016_Sentinel/B04_20m.jp2',
                         '/home/joe/Code/IceSurfClassifiers/2016_Sentinel/B05_20m.jp2',
                         '/home/joe/Code/IceSurfClassifiers/2016_Sentinel/B06_20m.jp2',
                         '/home/joe/Code/IceSurfClassifiers/2016_Sentinel/B07_20m.jp2',
                         '/home/joe/Code/IceSurfClassifiers/2016_Sentinel/B8A_20m.jp2',
                         '/home/joe/Code/IceSurfClassifiers/2016_Sentinel/B11_20m.jp2',
                         '/home/joe/Code/IceSurfClassifiers/2016_Sentinel/B12_20m.jp2']
    elif year == 2017:

        Sentinel_jp2s = ['/home/joe/Code/IceSurfClassifiers/2017_Sentinel/B02_20m.jp2',
                         '/home/joe/Code/IceSurfClassifiers/2017_Sentinel/B03_20m.jp2',
                         '/home/joe/Code/IceSurfClassifiers/2017_Sentinel/B04_20m.jp2',
                         '/home/joe/Code/IceSurfClassifiers/2017_Sentinel/B05_20m.jp2',
                         '/home/joe/Code/IceSurfClassifiers/2017_Sentinel/B06_20m.jp2',
                         '/home/joe/Code/IceSurfClassifiers/2017_Sentinel/B07_20m.jp2',
                         '/home/joe/Code/IceSurfClassifiers/2017_Sentinel/B8A_20m.jp2',
                         '/home/joe/Code/IceSurfClassifiers/2017_Sentinel/B11_20m.jp2',
                         '/home/joe/Code/IceSurfClassifiers/2017_Sentinel/B12_20m.jp2']
    else:

        print('ERROR: PLEASE CHOOSE TO USE IMAGERY FROM EITHER 2016 or 2017')

    # Read in raw HCRF data to DataFrame. This version pulls in HCRF data from 2016 and 2017

    hcrf_master = pd.read_csv(HCRF_file)
    HA_hcrf = pd.DataFrame()
    LA_hcrf = pd.DataFrame()
    CI_hcrf = pd.DataFrame()
    CC_hcrf = pd.DataFrame()
    WAT_hcrf = pd.DataFrame()
    SN_hcrf = pd.DataFrame()

    # Group site names according to surface class

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
               '5_8_16_site3_ice5', '5_8_16_site3_ice6', '5_8_16_site3_ice7', '5_8_16_site3_ice8',
               '5_8_16_site3_ice9']

    LAsites = ['14_7_S2', '14_7_S3', '14_7_SB2', '14_7_SB3', '14_7_SB7', '15_7_S2',
               '15_7_SB4', '20_7_SB1', '20_7_SB3', '21_7_S1', '21_7_S5', '21_7_SB4', '22_7_SB2',
               '22_7_SB3', '22_7_S1', '23_7_S1', '23_7_S2', '24_7_S2', 'MA_1', 'MA_2', 'MA_3',
               'MA_5', 'MA_6', 'MA_8', 'MA_9', 'MA_10', 'MA_12', 'MA_13', 'MA_16', 'MA_19',
               '13_7_S1', '13_7_S3', '14_7_S1', '15_7_S1', '15_7_SB2', '20_7_SB2', '21_7_SB5',
               '21_7_SB8', '25_7_S3', '5_8_16_site2_ice10', '5_8_16_site2_ice5',
               '5_8_16_site2_ice9', '27_7_16_SITE3_WHITE3']

    CIsites = ['21_7_S4', '13_7_SB3', '15_7_S4', '15_7_SB1', '15_7_SB5', '21_7_S2',
               '21_7_SB3', '22_7_S2', '22_7_S4', '23_7_SB1', '23_7_SB2', '23_7_S4',
               'WI_1', 'WI_2', 'WI_4', 'WI_5', 'WI_6', 'WI_7', 'WI_9', 'WI_10', 'WI_11',
               'WI_12', 'WI_13', '27_7_16_SITE3_WHITE1', '27_7_16_SITE3_WHITE2',
               '27_7_16_SITE2_ICE2', '27_7_16_SITE2_ICE4', '27_7_16_SITE2_ICE6',
               '5_8_16_site2_ice1', '5_8_16_site2_ice2', '5_8_16_site2_ice3',
               '5_8_16_site2_ice4', '5_8_16_site2_ice6', '5_8_16_site2_ice8',
               '5_8_16_site3_ice1', '5_8_16_site3_ice4']

    CCsites = ['DISP1', 'DISP2', 'DISP3', 'DISP4', 'DISP5', 'DISP6', 'DISP7', 'DISP8',
               'DISP9', 'DISP10', 'DISP11', 'DISP12', 'DISP13', 'DISP14', '27_7_16_SITE3_DISP1',
               '27_7_16_SITE3_DISP3']

    WATsites = ['21_7_SB5', '21_7_SB8', 'WAT_1', 'WAT_3', 'WAT_6']

    SNsites = ['14_7_S4', '14_7_SB6', '14_7_SB8', '17_7_SB2', 'SNICAR100', 'SNICAR200',
               'SNICAR300', 'SNICAR400', 'SNICAR500', 'SNICAR600', 'SNICAR700', 'SNICAR800',
               'SNICAR900', 'SNICAR1000', '27_7_16_KANU_', '27_7_16_SITE2_1',
               '5_8_16_site1_snow10', '5_8_16_site1_snow2', '5_8_16_site1_snow3',
               '5_8_16_site1_snow4', '5_8_16_site1_snow6', '5_8_16_site1_snow7',
               '5_8_16_site1_snow9']

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

        # plot spectra

    if plot_spectra:

        WL = np.arange(350, 2501, 1)

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

    # Make dataframe with column for label, columns for reflectance at key wavelengths
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

    R = pd.DataFrame()
    R['R140'] = np.array(SN_hcrf.iloc[140])
    R['R210'] = np.array(SN_hcrf.iloc[210])
    R['R315'] = np.array(SN_hcrf.iloc[315])
    R['R355'] = np.array(SN_hcrf.iloc[355])
    R['R390'] = np.array(SN_hcrf.iloc[390])
    R['R433'] = np.array(SN_hcrf.iloc[433])
    R['R515'] = np.array(SN_hcrf.iloc[515])
    R['R1260'] = np.array(SN_hcrf.iloc[1260])
    R['R1840'] = np.array(SN_hcrf.iloc[1840])

    R['label'] = 'SN'

    X = X.append(Y, ignore_index=True)
    X = X.append(Z, ignore_index=True)
    X = X.append(P, ignore_index=True)
    X = X.append(Q, ignore_index=True)
    X = X.append(R, ignore_index=True)

    # Create features and labels (XX = features - all data but no labels, YY = labels only)
    XX = X.drop(['label'], 1)
    YY = X['label']
    XX.head()  # print the top 5 rows of dataframe to console

    return Sentinel_jp2s, X, XX, YY



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

    print('Random Forest accuracy', accuracy_RF, '\nRandom Forest F1 Score = ', f1_RF, '\nRandom Forest Recall',
          recall_RF, '\nRandom Forest Precision = ', precision_RF)

    return clf



def ClassifyImages(Sentinel_jp2s, clf, plot_maps = True, savefigs=False):

    startTime = datetime.now()

    # Import multispectral imagery from Sentinel 2 and apply ML algorithm to classify surface

    jp2s = Sentinel_jp2s
    arrs = []
    for jp2 in jp2s:
        with rasterio.open(jp2) as f:
            arrs.append(f.read(1))

    data = np.array(arrs, dtype=arrs[0].dtype)

    # get dimensions of each band layer
    lenx, leny = np.shape(data[0])

    # convert image bands into single 5-dimensional numpy array
    test_array = np.array([data[0] / 10000, data[1] / 10000, data[2] / 10000, data[3] / 10000, data[4] / 10000,
                           data[5] / 10000, data[6] / 10000, data[7] / 10000, data[8] / 10000])
    test_array = test_array.reshape(9, lenx * leny)  # reshape into 5 x 1D arrays
    test_array = test_array.transpose()  # transpose so that bands are read as features

    # create albedo array by applying Knap (1999) narrowband - broadband conversion
    albedo_array = np.array([0.356 * (data[0] / 10000) + 0.13 * (data[2] / 10000) + 0.373 * (
                data[6] / 10000) + 0.085 * (data[7] / 10000) + 0.072 * (data[8] / 10000) - 0.0018])

    # apply ML algorithm to 4-value array for each pixel - predict surface type
    predicted = clf.predict(test_array)
    predicted = np.array(predicted)

    # convert surface class (string) to a numeric value for plotting
    predicted[predicted == 'SN'] = float(1)
    predicted[predicted == 'WAT'] = float(2)
    predicted[predicted == 'CC'] = float(3)
    predicted[predicted == 'CI'] = float(4)
    predicted[predicted == 'LA'] = float(5)
    predicted[predicted == 'HA'] = float(6)

    # ensure array data type is float (required for imshow)
    predicted = predicted.astype(float)

    # reshape 1D array back into original image dimensions
    predicted = np.reshape(predicted, [lenx, leny])
    albedo_array = np.reshape(albedo_array, [lenx, leny])

    # split image into 3 ice-covered areas that together represent majority of glaciated land in image (avoiding
    # ice-free areas of satellite image) NB a mask isolating the ice over the entire img would be a better way to
    # do this.

    predicted1 = predicted[2170:2975, 2130:4350]
    predicted2 = predicted[1400:2160, 2630:4740]
    predicted3 = predicted[0:1405, 2800:5075]
    albedo_array1 = albedo_array[2170:2975, 2130:4350]
    albedo_array2 = albedo_array[1400:2160, 2630:4740]
    albedo_array3 = albedo_array[0:1405, 2800:5075]

    cmap1 = mpl.colors.ListedColormap(['white', 'slategray', 'black', 'lightsteelblue', 'gold', 'orangered'])
    cmap2 = 'Greys_r'

    # plot classified surface

    if plot_maps:

        plt.figure(figsize=(30,30))
        plt.title("3 Classified ice surfaces and their albedos: SW Greenland Ice Sheet", fontsize = 28)

        plt.subplot(321)
        plt.imshow(predicted1, cmap=cmap1), plt.grid(None), plt.colorbar(), plt.title("Area 1: Classified")

        plt.subplot(322)
        plt.imshow(albedo_array1, cmap=cmap2), plt.grid(None), plt.colorbar(), plt.title("Area 1: Albedo")

        plt.subplot(323)
        plt.imshow(predicted2, cmap=cmap1), plt.grid(None), plt.colorbar(), plt.title("Area 2: Classified")

        plt.subplot(324)
        plt.imshow(albedo_array2, cmap=cmap2), plt.grid(None), plt.colorbar(), plt.title("Area 2: Albedo")

        plt.subplot(325)
        plt.imshow(predicted3, cmap=cmap1), plt.grid(None), plt.colorbar(), plt.title("Area 3 : Classified")

        plt.subplot(326)
        plt.imshow(albedo_array3, cmap=cmap2), plt.grid(None), plt.colorbar(), plt.title("Area 3: Albedo")

        if not savefigs:
            plt.show()


    if savefigs:

        plt.savefig(str(savefig_path + "confusion_matrices.jpg"), dpi=300)
        plt.show()


    print("\nTime taken to classify image = ", datetime.now() - startTime)

    return predicted1, predicted2, predicted3, albedo_array1, albedo_array2, albedo_array3



def CoverageStats(predicted1, predicted2, predicted3):

    res = 0.02  # Ground resolution of sentinel data in km
    counter = 0

    for i in predicted1, predicted2, predicted3:
        counter += 1

        x, y = np.shape(i)
        area_of_image = (x * res) * (y * res)  # area of selected region

        # Calculate coverage stats
        numHA = (i == 6).sum()
        numLA = (i == 5).sum()
        numCI = (i == 4).sum()
        numCC = (i == 3).sum()
        numWAT = (i == 2).sum()
        numSN = (i == 1).sum()
        total_pix = numHA + numLA + numCI + numCC + numWAT + numSN

        tot_alg_coverage = (numHA + numLA) / total_pix * 100
        HA_coverage = (numHA) / total_pix * 100
        LA_coverage = (numLA) / total_pix * 100
        CI_coverage = (numCI) / total_pix * 100
        CC_coverage = (numCC) / total_pix * 100
        WAT_coverage = (numWAT) / total_pix * 100
        SN_coverage = (numSN) / total_pix * 100

        # Print coverage summary
        print()
        print('**** SUMMARY ****')
        print('{} Area of image = '.format(counter), area_of_image, 'km')
        print('{}  % algal coverage (Hbio + Lbio) = '.format(counter), np.round(tot_alg_coverage, 2))
        print('{}  % Hbio coverage = '.format(counter), np.round(HA_coverage, 2))
        print('{}  % Lbio coverage = '.format(counter), np.round(LA_coverage, 2))
        print('{}  % cryoconite coverage = '.format(counter), np.round(CC_coverage, 2))
        print('{}  % clean ice coverage = '.format(counter), np.round(CI_coverage, 2))
        print('{}  % water coverage = '.format(counter), np.round(WAT_coverage, 2))
        print('{}  % snow coverage = '.format(counter), np.round(SN_coverage, 2))
        print()
        print()

    return


def albedo_report_by_site(predicted1, predicted2, predicted3, albedo_array1, albedo_array2, albedo_array3):
    counter = 0

    albedo_DF1 = pd.DataFrame()
    albedo_DF2 = pd.DataFrame()
    albedo_DF3 = pd.DataFrame()
    HA_DF1 = pd.DataFrame()
    LA_DF1 = pd.DataFrame()
    WAT_DF1 = pd.DataFrame()
    CI_DF1 = pd.DataFrame()
    CC_DF1 = pd.DataFrame()
    SN_DF1 = pd.DataFrame()
    HA_DF2 = pd.DataFrame()
    LA_DF2 = pd.DataFrame()
    WAT_DF2 = pd.DataFrame()
    CI_DF2 = pd.DataFrame()
    CC_DF2 = pd.DataFrame()
    SN_DF2 = pd.DataFrame()
    HA_DF3 = pd.DataFrame()
    LA_DF3 = pd.DataFrame()
    WAT_DF3 = pd.DataFrame()
    CI_DF3 = pd.DataFrame()
    CC_DF3 = pd.DataFrame()
    SN_DF3 = pd.DataFrame()

    for i in predicted1, predicted2, predicted3:

        counter += 1

        PP = i.ravel()
        print(len(PP))

        if counter == 1:

            AA = albedo_array1.ravel()
            albedo_DF1['class'] = PP
            albedo_DF1['albedo'] = AA

            # divide albedo dataframe into individual classes for summary stats. include only
            # rows where albedo is between 0.05 and 0.95 percentiles to remove outliers

            HA_DF1 = albedo_DF1[albedo_DF1['class'] == 6]
            HA_DF1 = HA_DF1[HA_DF1['albedo'] > HA_DF1['albedo'].quantile(0.05)]
            HA_DF1 = HA_DF1[HA_DF1['albedo'] < HA_DF1['albedo'].quantile(0.95)]

            LA_DF1 = albedo_DF1[albedo_DF1['class'] == 5]
            LA_DF1 = LA_DF1[LA_DF1['albedo'] > LA_DF1['albedo'].quantile(0.05)]
            LA_DF1 = LA_DF1[LA_DF1['albedo'] < LA_DF1['albedo'].quantile(0.95)]

            CI_DF1 = albedo_DF1[albedo_DF1['class'] == 4]
            CI_DF1 = CI_DF1[CI_DF1['albedo'] > CI_DF1['albedo'].quantile(0.05)]
            CI_DF1 = CI_DF1[CI_DF1['albedo'] < CI_DF1['albedo'].quantile(0.95)]

            CC_DF1 = albedo_DF1[albedo_DF1['class'] == 3]
            CC_DF1 = CC_DF1[CC_DF1['albedo'] > CC_DF1['albedo'].quantile(0.05)]
            CC_DF1 = CC_DF1[CC_DF1['albedo'] < CC_DF1['albedo'].quantile(0.95)]

            WAT_DF1 = albedo_DF1[albedo_DF1['class'] == 2]
            WAT_DF1 = WAT_DF1[WAT_DF1['albedo'] > WAT_DF1['albedo'].quantile(0.05)]
            WAT_DF1 = WAT_DF1[WAT_DF1['albedo'] < WAT_DF1['albedo'].quantile(0.95)]

            SN_DF1 = albedo_DF1[albedo_DF1['class'] == 1]
            SN_DF1 = SN_DF1[SN_DF1['albedo'] > SN_DF1['albedo'].quantile(0.05)]
            SN_DF1 = SN_DF1[SN_DF1['albedo'] < SN_DF1['albedo'].quantile(0.95)]

            # Calculate summary stats
            mean_CC1 = CC_DF1['albedo'].mean()
            std_CC1 = CC_DF1['albedo'].std()
            max_CC1 = CC_DF1['albedo'].max()
            min_CC1 = CC_DF1['albedo'].min()

            mean_CI1 = CI_DF1['albedo'].mean()
            std_CI1 = CI_DF1['albedo'].std()
            max_CI1 = CI_DF1['albedo'].max()
            min_CI1 = CI_DF1['albedo'].min()

            mean_LA1 = LA_DF1['albedo'].mean()
            std_LA1 = LA_DF1['albedo'].std()
            max_LA1 = LA_DF1['albedo'].max()
            min_LA1 = LA_DF1['albedo'].min()

            mean_HA1 = HA_DF1['albedo'].mean()
            std_HA1 = HA_DF1['albedo'].std()
            max_HA1 = HA_DF1['albedo'].max()
            min_HA1 = HA_DF1['albedo'].min()

            mean_WAT1 = WAT_DF1['albedo'].mean()
            std_WAT1 = WAT_DF1['albedo'].std()
            max_WAT1 = WAT_DF1['albedo'].max()
            min_WAT1 = WAT_DF1['albedo'].min()

            mean_SN1 = SN_DF1['albedo'].mean()
            std_SN1 = SN_DF1['albedo'].std()
            max_SN1 = SN_DF1['albedo'].max()
            min_SN1 = SN_DF1['albedo'].min()

            print()
            print('*** Albedo Stats 1 ***')
            print()
            print('mean albedo WAT 1 = ', mean_WAT1)
            print('mean albedo CC 1 = ', mean_CC1)
            print('mean albedo CI 1 = ', mean_CI1)
            print('mean albedo LA 1 = ', mean_LA1)
            print('mean albedo HA 1 = ', mean_HA1)
            print('mean albedo SN 1 = ', mean_SN1)
            print('n HA 1 = ', len(HA_DF1))
            print('n LA 1 = ', len(LA_DF1))
            print('n CI 1 = ', len(CI_DF1))
            print('n CC 1 = ', len(CC_DF1))
            print('n WAT 1 = ', len(WAT_DF1))
            print('n SN 1 = ', len(SN_DF1))

        elif counter == 2:

            AA = albedo_array2.ravel()
            albedo_DF2['class'] = PP
            albedo_DF2['albedo'] = AA

            HA_DF2 = albedo_DF2[albedo_DF2['class'] == 6]
            HA_DF2 = HA_DF2[HA_DF2['albedo'] > HA_DF2['albedo'].quantile(0.05)]
            HA_DF2 = HA_DF2[HA_DF2['albedo'] < HA_DF2['albedo'].quantile(0.95)]

            LA_DF2 = albedo_DF2[albedo_DF2['class'] == 5]
            LA_DF2 = LA_DF2[LA_DF2['albedo'] > LA_DF2['albedo'].quantile(0.05)]
            LA_DF2 = LA_DF2[LA_DF2['albedo'] < LA_DF2['albedo'].quantile(0.95)]

            CI_DF2 = albedo_DF2[albedo_DF2['class'] == 4]
            CI_DF2 = CI_DF2[CI_DF2['albedo'] > CI_DF2['albedo'].quantile(0.05)]
            CI_DF2 = CI_DF2[CI_DF2['albedo'] < CI_DF2['albedo'].quantile(0.95)]

            CC_DF2 = albedo_DF2[albedo_DF2['class'] == 3]
            CC_DF2 = CC_DF2[CC_DF2['albedo'] > CC_DF2['albedo'].quantile(0.05)]
            CC_DF2 = CC_DF2[CC_DF2['albedo'] < CC_DF2['albedo'].quantile(0.95)]

            WAT_DF2 = albedo_DF2[albedo_DF2['class'] == 2]
            WAT_DF2 = WAT_DF2[WAT_DF2['albedo'] > WAT_DF2['albedo'].quantile(0.05)]
            WAT_DF2 = WAT_DF2[WAT_DF2['albedo'] < WAT_DF2['albedo'].quantile(0.95)]

            SN_DF2 = albedo_DF2[albedo_DF2['class'] == 1]
            SN_DF2 = SN_DF2[SN_DF2['albedo'] > SN_DF2['albedo'].quantile(0.05)]
            SN_DF2 = SN_DF2[SN_DF2['albedo'] < SN_DF2['albedo'].quantile(0.95)]

            # Calculate summary stats
            mean_CC2 = CC_DF2['albedo'].mean()
            std_CC2 = CC_DF2['albedo'].std()
            max_CC2 = CC_DF2['albedo'].max()
            min_CC2 = CC_DF2['albedo'].min()

            mean_CI2 = CI_DF2['albedo'].mean()
            std_CI2 = CI_DF2['albedo'].std()
            max_CI2 = CI_DF2['albedo'].max()
            min_CI2 = CI_DF2['albedo'].min()

            mean_LA2 = LA_DF2['albedo'].mean()
            std_LA2 = LA_DF2['albedo'].std()
            max_LA2 = LA_DF2['albedo'].max()
            min_LA2 = LA_DF2['albedo'].min()

            mean_HA2 = HA_DF2['albedo'].mean()
            std_HA2 = HA_DF2['albedo'].std()
            max_HA2 = HA_DF2['albedo'].max()
            min_HA2 = HA_DF2['albedo'].min()

            mean_WAT2 = WAT_DF2['albedo'].mean()
            std_WAT2 = WAT_DF2['albedo'].std()
            max_WAT2 = WAT_DF2['albedo'].max()
            min_WAT2 = WAT_DF2['albedo'].min()

            mean_SN2 = SN_DF2['albedo'].mean()
            std_SN2 = SN_DF2['albedo'].std()
            max_SN2 = SN_DF2['albedo'].max()
            min_SN2 = SN_DF2['albedo'].min()

            print()
            print('*** Albedo Stats 2 ***')
            print()
            print('mean albedo WAT 2 = ', mean_WAT2)
            print('mean albedo CC 2 = ', mean_CC2)
            print('mean albedo CI 2 = ', mean_CI2)
            print('mean albedo LA 2 = ', mean_LA2)
            print('mean albedo HA 2 = ', mean_HA2)
            print('mean albedo SN 2 = ', mean_SN2)
            print('n HA 2 = ', len(HA_DF2))
            print('n LA 2 = ', len(LA_DF2))
            print('n CI 2 = ', len(CI_DF2))
            print('n CC 2 = ', len(CC_DF2))
            print('n WAT 2 = ', len(WAT_DF2))
            print('n SN 2 = ', len(SN_DF2))

        elif counter == 3:

            AA = albedo_array3.ravel()
            albedo_DF3['class'] = PP
            albedo_DF3['albedo'] = AA

            HA_DF3 = albedo_DF3[albedo_DF3['class'] == 6]
            HA_DF3 = HA_DF3[HA_DF3['albedo'] > HA_DF3['albedo'].quantile(0.05)]
            HA_DF3 = HA_DF3[HA_DF3['albedo'] < HA_DF3['albedo'].quantile(0.95)]

            LA_DF3 = albedo_DF3[albedo_DF3['class'] == 5]
            LA_DF3 = LA_DF3[LA_DF3['albedo'] > LA_DF3['albedo'].quantile(0.05)]
            LA_DF3 = LA_DF3[LA_DF3['albedo'] < LA_DF3['albedo'].quantile(0.95)]

            CI_DF3 = albedo_DF3[albedo_DF3['class'] == 4]
            CI_DF3 = CI_DF3[CI_DF3['albedo'] > CI_DF3['albedo'].quantile(0.05)]
            CI_DF3 = CI_DF3[CI_DF3['albedo'] < CI_DF3['albedo'].quantile(0.95)]

            CC_DF3 = albedo_DF3[albedo_DF3['class'] == 3]
            CC_DF3 = CC_DF3[CC_DF3['albedo'] > CC_DF3['albedo'].quantile(0.05)]
            CC_DF3 = CC_DF3[CC_DF3['albedo'] < CC_DF3['albedo'].quantile(0.95)]

            WAT_DF3 = albedo_DF3[albedo_DF3['class'] == 2]
            WAT_DF3 = WAT_DF3[WAT_DF3['albedo'] > WAT_DF3['albedo'].quantile(0.05)]
            WAT_DF3 = WAT_DF3[WAT_DF3['albedo'] < WAT_DF3['albedo'].quantile(0.95)]

            SN_DF3 = albedo_DF3[albedo_DF3['class'] == 1]
            SN_DF3 = SN_DF3[SN_DF3['albedo'] > SN_DF3['albedo'].quantile(0.05)]
            SN_DF3 = SN_DF3[SN_DF3['albedo'] < SN_DF3['albedo'].quantile(0.95)]

            # Calculate summary stats
            mean_CC3 = CC_DF3['albedo'].mean()
            std_CC3 = CC_DF3['albedo'].std()
            max_CC3 = CC_DF3['albedo'].max()
            min_CC3 = CC_DF3['albedo'].min()

            mean_CI3 = CI_DF3['albedo'].mean()
            std_CI3 = CI_DF3['albedo'].std()
            max_CI3 = CI_DF3['albedo'].max()
            min_CI3 = CI_DF3['albedo'].min()

            mean_LA3 = LA_DF3['albedo'].mean()
            std_LA3 = LA_DF3['albedo'].std()
            max_LA3 = LA_DF3['albedo'].max()
            min_LA3 = LA_DF3['albedo'].min()

            mean_HA3 = HA_DF3['albedo'].mean()
            std_HA3 = HA_DF3['albedo'].std()
            max_HA3 = HA_DF3['albedo'].max()
            min_HA3 = HA_DF3['albedo'].min()

            mean_WAT3 = WAT_DF3['albedo'].mean()
            std_WAT3 = WAT_DF3['albedo'].std()
            max_WAT3 = WAT_DF3['albedo'].max()
            min_WAT3 = WAT_DF3['albedo'].min()

            mean_SN3 = SN_DF3['albedo'].mean()
            std_SN3 = SN_DF3['albedo'].std()
            max_SN3 = SN_DF3['albedo'].max()
            min_SN3 = SN_DF3['albedo'].min()

            print()
            print('*** Albedo Stats 3 ***')
            print()
            print('mean albedo WAT 3 = ', mean_WAT3)
            print('mean albedo CC 3 = ', mean_CC3)
            print('mean albedo CI 3 = ', mean_CI3)
            print('mean albedo LA 3 = ', mean_LA3)
            print('mean albedo HA 3 = ', mean_HA3)
            print('mean albedo SN 3 = ', mean_SN3)
            print('n HA 3 = ', len(HA_DF3))
            print('n LA 3 = ', len(LA_DF3))
            print('n CI 3 = ', len(CI_DF3))
            print('n CC 3 = ', len(CC_DF3))
            print('n WAT 3 = ', len(WAT_DF3))
            print('n SN 3 = ', len(SN_DF3))

        albedo_DF1.to_csv('2016Sentinel_20m_albedo_dataset_Area1.csv')
        albedo_DF2.to_csv('2016Sentinel_20m_albedo_dataset_Area2.csv')
        albedo_DF3.to_csv('2016Sentinel_20m_albedo_dataset_Area3.csv')
        albedo_DFall = pd.concat([albedo_DF1, albedo_DF2, albedo_DF3])
        albedo_DFall.to_csv('2016Sentinel_20m_albedo_dataset_allsites.csv')

    return albedo_DF1, albedo_DF2, albedo_DF3, albedo_DFall, HA_DF1, LA_DF1, CI_DF1, CC_DF1, WAT_DF1, SN_DF1, HA_DF2,\
           LA_DF2, CI_DF2, CC_DF2, WAT_DF2, SN_DF2, HA_DF3, LA_DF3, CI_DF3, CC_DF3, WAT_DF3, SN_DF3




def albedo_report_all_sites(albedo_DFall, HA_DF1, LA_DF1, CI_DF1, CC_DF1, WAT_DF1, SN_DF1, HA_DF2, LA_DF2, CI_DF2,
                            CC_DF2, WAT_DF2, SN_DF2, HA_DF3, LA_DF3, CI_DF3, CC_DF3, WAT_DF3, SN_DF3):

    HA_DF = pd.concat([HA_DF1, HA_DF2, HA_DF3])
    LA_DF = pd.concat([LA_DF1, LA_DF2, LA_DF3])
    CI_DF = pd.concat([CI_DF1, CI_DF2, CI_DF3])
    CC_DF = pd.concat([CC_DF1, CC_DF2, CC_DF3])
    WAT_DF = pd.concat([WAT_DF1, WAT_DF2, WAT_DF3])
    SN_DF = pd.concat([SN_DF1, SN_DF2, SN_DF3])

    print('**SUMMARY FOR ALL SITES COMBINED ***')
    print()
    print('Mean Albedo HA: ', HA_DF['albedo'].mean())
    print('Std Albedo HA: ', HA_DF['albedo'].std())
    print('Min Albedo HA: ', HA_DF['albedo'].min())
    print('Max Albedo HA: ', HA_DF['albedo'].max())
    print('HA number of observations (n) = ', len(HA_DF))
    print()
    print('Mean Albedo LA: ', LA_DF['albedo'].mean())
    print('Std Albedo LA: ', LA_DF['albedo'].std())
    print('Min Albedo LA: ', LA_DF['albedo'].min())
    print('Max Albedo LA: ', LA_DF['albedo'].max())
    print('LA number of observations (n) = ', len(LA_DF))
    print()
    print('Mean Albedo CI: ', CI_DF['albedo'].mean())
    print('Std Albedo CI: ', CI_DF['albedo'].std())
    print('Min Albedo CI: ', CI_DF['albedo'].min())
    print('Max Albedo CI: ', CI_DF['albedo'].max())
    print('CI number of observations (n) = ', len(CI_DF))
    print()
    print('Mean Albedo CC: ', CC_DF['albedo'].mean())
    print('Std Albedo CC: ', CC_DF['albedo'].std())
    print('Min Albedo CC: ', CC_DF['albedo'].min())
    print('Max Albedo CC: ', CC_DF['albedo'].max())
    print('CC number of observations (n) = ', len(CC_DF))
    print()
    print('Mean Albedo WAT: ', WAT_DF['albedo'].mean())
    print('Std Albedo WAT: ', WAT_DF['albedo'].std())
    print('Min Albedo WAT: ', WAT_DF['albedo'].min())
    print('Max Albedo WAT: ', WAT_DF['albedo'].max())
    print('WAT number of observations (n) = ', len(WAT_DF))
    print()
    print('Mean Albedo SN: ', SN_DF['albedo'].mean())
    print('Std Albedo SN: ', SN_DF['albedo'].std())
    print('Min Albedo SN: ', SN_DF['albedo'].min())
    print('Max Albedo SN: ', SN_DF['albedo'].max())
    print('SN number of observations (n) = ', len(SN_DF))

    return HA_DF, LA_DF, CI_DF, CC_DF, WAT_DF, SN_DF





# RUN FUNCTIONS

# create dataset
Sentinel_jp2s, X, XX, YY = create_dataset(HCRF_file, year=2016, plot_spectra=False, savefigs=False)

#optimise and train model
clf = optimise_train_model(X, XX, YY, test_size=0.3)

# apply model to Sentinel2 image
predicted1,predicted2,predicted3,albedo_array1,albedo_array2,albedo_array3 =  ClassifyImages(Sentinel_jp2s,clf,
plot_maps = False, savefigs=False)

# calculate coverage stats for each sub-area
CoverageStats(predicted1,predicted2,predicted3)

# obtain albedo summary stats
albedo_DF1, albedo_DF2, albedo_DF3, albedo_DFall, HA_DF1, LA_DF1, CI_DF1, CC_DF1, WAT_DF1, SN_DF1, HA_DF2, \
LA_DF2, CI_DF2, CC_DF2, WAT_DF2, SN_DF2, HA_DF3, LA_DF3, CI_DF3, CC_DF3, WAT_DF3, SN_DF3= \
albedo_report_by_site(predicted1,predicted2,predicted3,albedo_array1,albedo_array2,albedo_array3)

# obtain albedo stats for all sites combined
HA_DF,LA_DF,CI_DF,CC_DF,WAT_DF,SN_DF = albedo_report_all_sites(albedo_DFall, HA_DF1, LA_DF1, CI_DF1, CC_DF1, WAT_DF1,
SN_DF1, HA_DF2, LA_DF2, CI_DF2, CC_DF2, WAT_DF2, SN_DF2, HA_DF3, LA_DF3, CI_DF3, CC_DF3, WAT_DF3, SN_DF3)