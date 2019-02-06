"""
*** INFO ***

code written by Joseph Cook (University of Sheffield), 2018. Correspondence to joe.cook@sheffield.ac.uk

*** OVERVIEW ***

This code trains a random forest classifier on ground-level reflectance data, then deploys it to classify Sentinel 2
imagery into various surface categories:

Water, Snow, Clean Ice, Cryoconite, Light Algal Bloom, Heavy Algal Bloom

The result is a classified ice surface map. The coverage statistics are calculated and reported.
The albedo of each pixel is calculated from the multispectral reflectance using Liang et al's (2002) narrowband to
broadband conversion formula. The result is an albedo map of the ice surface.

Both the classified map and albedo map are trimmed to remove non-ice areas using the Greenland Ice Mapping Project mask
before spatial stats are calculated.


*** PREREQUISITES ***

1) The core training data is saved as a csv named "HRCF_master_machine_snicar.csv". This is a collection of spectral
reflectance measurements made at ground level using an ASD Field Spec Pro 3 for surfaces whose characteristics are
known.

2) Sentinel-2 band images. Folders containing Level 1C products can be downloaded from Earthexplorer.usgs.gov
These must be converted from level 1C to Level 2A (i.e. corrected for atmospheric effects and reprojected to a
consistent 20m ground resolution) using the ESA command line tool Sen2Cor.

This requires downloading the Sen2Cor software and running from the command line. Instructions are available here:
https://forum.step.esa.int/t/sen2cor-2-4-0-stand-alone-installers-how-to-install/6908

Sen2Cor details:

L2A processor path =  '/home/joe/Sen2Cor/Sen2Cor-02.05.05-Linux64/bin/L2A_Process'
Default configuration file = '/home/joe/sen2cor/2.5/cfg/L2A_GIPP.xml'

With file downloaded from EarthExplorer on desktop, L1C to L2A processing achieved using command:

>> /home/joe/Sen2Cor/Sen2Cor-02.05.05-Linux64/bin/L2A_Process ...
>> /home/joe/Desktop/S2A_MSIL1C_20160721T151912_N0204_R068_T22WEV_20160721T151913.SAFE

Then reformat the jp2 images for each band into netCDF files using gdal:

>> source activate IceSurfClassifiers
>> cd /home/joe/Desktop/S2A_MSIL2A_20160721T151912_N0204_R068_T22WEV_20160721T151913.SAFE/GRANULE/L2A_T22WEV_A005642_
   20160721T151913/IMG_DATA/R20m/

>> gdal_translate L2A_T22WEV_20160721T151912_B02_20m.jp2 /home/joe/Desktop/S2A_NetCDFs/B02.nc

repeat for each band. The final processed files are then available in the desktop folder 'S2A_NetCDFs/Site/' and saved
as B02.nc, B03.nc etc. These netcdfs are then used as input data in this script.

The resulting NetCDF files are then used as input data for this script.

3) The GIMP mask downloaded from https://nsidc.org/data/nsidc-0714/versions/1 must be saved to the working directory.
Ensure the downloaded tile is the correct one for the section of ice sheet being examined.


*** FUNCTIONS***

This script is divided into several functions. The first function (create_dataset) preprocesses the raw data into a
format appropriate for supervised classification (i.e. explicit features and labels). The raw hyperspectral data is
first organised into separate pandas dataframes for each surface class. The data is then reduced down to the reflectance
at the nine key wavelengths coincident with those of the Sentinel 2 spectrometer. The remaining data are discarded.
The dataset is then arranged into columns with one column per wavelength and a separate column for the surface label.
The dataset's features are the reflectance at each wavelength, and the labels are the surface types.
The dataframes for each surface type are merged into one large dataframe and then the labels are removed and saved as a
separate dataframe. No scaling of the data is required because the reflectance is already normalised between 0 and 1 by
the spectrometer.

The second function (train_test_split) separates the dataset into training and test sets and trains a random forest
classifier. Setting n_jobs = -1 ensures the training and prediction phases are distributed over all available processor
cores. The performance of the trained model is calculated and displayed. The classifier can optionally be saved to a
.pkl file, or loaded in from an external .pkl file rather than continually retraining on the fly.

The third function (format_mask) reprojects the GIMP mask to an identical coordinate system, pixel size and spatial
extent to the Sentinel 2 images and returns a Boolean numpy array that will later be used to mask out non-ice areas
of the classificed map and albedo map

The fourth function applies the trained classifier to the sentinel 2 images and masks out non-ice areas, then applies
Liang et al(2002) narrowband to broadband conversion formula, producing a NetCDF file containing all the data arrays and
metadata along with a plot of each map.

The final function calculates spatial statistics for the classified surface and albed maps.

"""


###########################################################################################
############################# IMPORT MODULES #########################################

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import xarray as xr
import seaborn as sn
from osgeo import gdal, osr
import georaster

# matplotlib settings: use ggplot style and turn interactive mode off so that plots can be saved and not shown (for
# rapidly processing multiple images later)

mpl.style.use('ggplot')
plt.ioff()

# DEFINE FUNCTIONS
def set_paths(virtual_machine = False):

    if not virtual_machine:
        HCRF_file = '/home/joe/Code/IceSurfClassifiers/Training_Data/HCRF_master_machine_snicar.csv'
        savefig_path = '//home/joe/Desktop/'
        img_path = '/home/joe/Desktop/S2A_NetCDFs/KGR/'

        # paths for format_mask()
        Sentinel_template = '/home/joe/Desktop/S2_L2A/S2_L2A_KGR/GRANULE/L2A_T22WEV_A005642_20160721T151913/IMG_DATA/R20m/L2A_T22WEV_20160721T151912_B02_20m.jp2'
        mask_in = '/home/joe/Desktop/GIMP_MASK.tif'
        mask_out = '/home/joe/Desktop/GIMP_MASK.nc'

    else:
        # Virtual Machine
        # paths for create_dataset()
        HCRF_file = '/home/tothepoles/PycharmProjects/IceSurfClassifiers/Training_Data/HCRF_master_machine_snicar.csv'
        savefig_path = '/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_Resources/Sentinel_Outputs/'
        img_path = '/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_Resources/S2A_NetCDFs/'

        # paths for format_mask()
        Sentinel_template = '/data/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_Resources/L2A_T22WEV_20160721T151912_B02_20m.jp2'
        mask_in = '/data/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_Resources/GIMP_MASK.tif'
        mask_out = '/data/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_Resources/GIMP_MASK.nc'

    return HCRF_file, savefig_path,img_path, Sentinel_template, mask_in, mask_out



def create_dataset(HCRF_file, img_path, plot_spectra=True, savefigs=True):
# Sentinel 2 dataset
# create 3D numpy array with dim1 = band, dims 2 and 3 = spatial x and y. Values are reflectance.
    S2vals = np.zeros([9,5490,5490])
    bands = ['02','03','04','05','05','07','08','11','12']
    for i in np.arange(0,len(bands),1):
        S2BX = xr.open_dataset(str(img_path+'B'+bands[i]+'.nc'))
        S2BXarr = S2BX.to_array()
        S2BXvals = np.array(S2BXarr.variable.values[1])
        S2BXvals = S2BXvals.astype(float)
        S2vals[i,:,:] = S2BXvals

    S2vals = S2vals/10000 # correct unit from S2 L2A data to reflectance between 0-1

# ground reflectance dataset
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


    Zero = pd.DataFrame()
    Zero['R140'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R210'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R315'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R355'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R390'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R433'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R515'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R1260'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R1840'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['label'] = 'Zero'


    X = X.append(Y, ignore_index=True)
    X = X.append(Z, ignore_index=True)
    X = X.append(P, ignore_index=True)
    X = X.append(Q, ignore_index=True)
    X = X.append(R, ignore_index=True)
    X = X.append(Zero, ignore_index=True)

    return S2vals, X

def split_train_test(X, test_size=0.2, n_trees= 64, print_conf_mx = True, plot_conf_mx = True, savefigs = False,
                     show_model_performance = True, pickle_model=False):

    XX = X.drop(['label'],axis=1)
    YY = X['label']
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(XX, YY, test_size=test_size)
    clf = RandomForestClassifier(n_estimators=n_trees, max_leaf_nodes=16, n_jobs=-1)
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
                   xticklabels=['NaN', 'Snow', 'Water', 'Cryoconite','Clean Ice', 'Light Algae', 'Heavy Algae'],
                   yticklabels=['NaN', 'Snow', 'Water', 'Cryoconite', 'Clean Ice', 'Light Algae', 'Heavy Algae'],
                   cbar_kws={"shrink": 0.4, 'label':'frequency'}, ax=ax1), ax1.tick_params(axis='both', rotation=45)
        ax1.set_title('Confusion Matrix'), ax1.set_aspect('equal')

        sn.heatmap(norm_conf_mx, annot=True, annot_kws={"size": 16}, cmap=plt.cm.gray,
                   xticklabels=['NaN', 'Snow', 'Water', 'Cryoconite', 'Clean Ice', 'Light Algae', 'Heavy Algae'],
                   yticklabels=['NaN', 'Snow', 'Water', 'Cryoconite', 'Clean Ice', 'Light Algae', 'Heavy Algae'],
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


def format_mask (Sentinel_template,mask_in,mask_out):
    """
    Function reprojects GIMP mask to dimensions, resolution and spatial coords of the S2 images, enabling
    Boolean masking of land-ice area.

    INPUTS:
    Sentinel = Sentinel image to use as projection template
    mask_in = file path to mask file
    mask_out = file path to save reprojected mask
    res = resolution to save mask

    OUTPUTS:
    reprojected mask in .tif format

    """
    mask = gdal.Open(mask_in)

    mask_proj = mask.GetProjection()
    mask_geotrans = mask.GetGeoTransform()
    data_type = mask.GetRasterBand(1).DataType
    n_bands = mask.RasterCount

    Sentinel = gdal.Open(Sentinel_template)

    Sentinel_proj = Sentinel.GetProjection()
    Sentinel_geotrans = Sentinel.GetGeoTransform()
    w = Sentinel.RasterXSize
    h = Sentinel.RasterYSize

    mask_filename = mask_out
    new_mask = gdal.GetDriverByName('NETCDF').Create(mask_filename,
                                                w, h, n_bands, data_type)
    new_mask.SetGeoTransform(Sentinel_geotrans)
    new_mask.SetProjection(Sentinel_proj)

    gdal.ReprojectImage( mask, new_mask, mask_proj,
                         Sentinel_proj, gdal.GRA_NearestNeighbour)
    new_mask = None  # Flush disk

    # open netCDF mask and extract values to numpy array
    maskxr = xr.open_dataset(mask_out)
    mask_array = np.array(maskxr.Band1.values)
    #replace nans with 0 to create binary numerical array
    nans = np.isnan(mask_array)
    mask_array[nans]=0

    return mask_array


def ClassifyImages(S2vals, clf, mask_array, plot_maps = True, savefigs=False, save_netcdf = False):

    # get dimensions of each band layer
    lenx, leny = np.shape(S2vals[0])

    # convert image bands into single 5-dimensional numpy array
    S2valsT = S2vals.reshape(9, lenx * leny)  # reshape into 5 x 1D arrays
    S2valsT = S2valsT.transpose()  # transpose so that bands are read as features


    # create albedo array by applying Knap (1999) narrowband - broadband conversion
    albedo_array = np.array([0.356 * (S2vals[0]) + 0.13 * (S2vals[2]) + 0.373 * (
            S2vals[6]) + 0.085 * (S2vals[7]) + 0.072 * (S2vals[8]) - 0.0018])

    # apply ML algorithm to 4-value array for each pixel - predict surface type
    predicted = clf.predict(S2valsT)
    predicted = np.array(predicted)

    # convert surface class (string) to a numeric value for plotting
    predicted[predicted == 'SN'] = float(1)
    predicted[predicted == 'WAT'] = float(2)
    predicted[predicted == 'CC'] = float(3)
    predicted[predicted == 'CI'] = float(4)
    predicted[predicted == 'LA'] = float(5)
    predicted[predicted == 'HA'] = float(6)
    predicted[predicted == 'Zero'] = float(0)

    # ensure array data type is float (required for imshow)
    predicted = predicted.astype(float)

    # reshape 1D array back into original image dimensions

    predicted = np.reshape(predicted, [lenx, leny])
    albedo = np.reshape(albedo_array, [lenx, leny])


    # apply GIMP mask to ignore non-ice surfaces
    predicted = np.ma.masked_where(mask_array==0, predicted)
    albedo = np.ma.masked_where(mask_array==0, albedo)

    # mask out areas not covered by Sentinel tile, but not excluded by GIMP mask
    predicted = np.ma.masked_where(predicted <=0, predicted)
    albedo = np.ma.masked_where(albedo <=0, albedo)

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

    # 2) Create associated lat/lon coordinates DataArrays using georaster (imports geo metadata without loading img)
    # see georaster docs at https: // media.readthedocs.org / pdf / georaster / latest / georaster.pdf
    S2 = georaster.SingleBandRaster('NETCDF:"%s":Band1' % (str(img_path+'B02.nc')),
                                    load_data=False)
    lon, lat = S2.coordinates(latlon=True)
    S2 = None  # close file
    S2 = xr.open_dataset((str(img_path+'B02.nc')), chunks={'x': 2000, 'y': 2000})
    coords_geo = {'y': S2['y'], 'x': S2['x']}
    S2 = None  # close file

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
    predictedxr = xr.DataArray(predicted, coords=coords_geo, dims=['y', 'x'])
    predictedxr.encoding = {'dtype': 'int16', 'zlib': True, '_FillValue': -9999}
    predictedxr.name = 'Surface Class'
    predictedxr.attrs['long_name'] = 'Surface classified using Random Forest'
    predictedxr.attrs['units'] = 'None'
    predictedxr.attrs[
        'key'] = 'Unknown:0; Snow:1; Water:2; Cryoconite:3; Clean Ice:4; Light Algae:5; Heavy Algae:6'
    predictedxr.attrs['grid_mapping'] = 'UTM'

    # add albedo map array and add metadata
    albedoxr = xr.DataArray(albedo, coords=coords_geo, dims=['y', 'x'])
    albedoxr.encoding = {'dtype': 'int16', 'scale_factor': 0, 'zlib': True, '_FillValue': -9999}
    albedoxr.name = 'Surface albedo computed after Knap et al. (1999) narrowband-to-broadband conversion'
    albedoxr.attrs['units'] = 'dimensionless'
    albedoxr.attrs['grid_mapping'] = 'UTM'

    # collate data arrays into a dataset
    dataset = xr.Dataset({

        'classified': (['x', 'y'], predictedxr),
        'albedo': (['x', 'y'], albedoxr),
        'mask': (['x','y'],mask_array),
        'Projection': proj_info,
        'longitude': (['x', 'y'], lon_array),
        'latitude': (['x', 'y'], lat_array)
    })

    # add metadata for dataset
    dataset.attrs['Conventions'] = 'CF-1.4'
    dataset.attrs['Author'] = 'Joseph Cook (University of Sheffield, UK)'
    dataset.attrs[
        'title'] = 'Classified surface and albedo maps produced from Sentinel-2 ' \
                   'imagery of the SW Greenland Ice Sheet'

    # Additional geo-referencing
    dataset.attrs['nx'] = len(dataset.x)
    dataset.attrs['ny'] = len(dataset.y)
    dataset.attrs['xmin'] = float(dataset.x.min())
    dataset.attrs['ymax'] = float(dataset.y.max())
    dataset.attrs['spacing'] = 20

    # NC conventions metadata for dimensions variables
    dataset.x.attrs['units'] = 'meters'
    dataset.x.attrs['standard_name'] = 'projection_x_coordinate'
    dataset.x.attrs['point_spacing'] = 'even'
    dataset.x.attrs['axis'] = 'x'

    dataset.y.attrs['units'] = 'meters'
    dataset.y.attrs['standard_name'] = 'projection_y_coordinate'
    dataset.y.attrs['point_spacing'] = 'even'
    dataset.y.attrs['axis'] = 'y'

    # save dataset to netcdf if requested
    if save_netcdf:
        dataset.to_netcdf(savefig_path + "Classification_and_Albedo_Data.nc")

    if plot_maps or savefigs:

        cmap1 = mpl.colors.ListedColormap(
            ['purple', 'white', 'royalblue', 'black', 'lightskyblue', 'mediumseagreen', 'darkgreen'])
        cmap1.set_under(color='white')  # make sure background is white
        cmap2 = plt.get_cmap('Greys_r')  # reverse greyscale for albedo
        cmap2.set_under(color='white')  # make sure background is white

        fig = plt.figure(figsize=(15, 30))

        # first subplot = classified map
        class_labels = ['Unknown', 'Snow', 'Water', 'Cryoconite', 'Clean Ice', 'Light Algae', 'Heavy Algae']
        ax1 = plt.subplot(211)
        img = plt.imshow(predicted, cmap=cmap1,vmin=0, vmax=7)
        cbar = fig.colorbar(mappable = img, ax=ax1, fraction=0.045)

        n_classes = len(class_labels)
        tick_locs = np.arange(0.5,len(class_labels),1)
        cbar.set_ticks(tick_locs)
        cbar.ax.set_yticklabels(class_labels, fontsize=26, rotation=0, va='center')
        cbar.set_label('Surface Class',fontsize=22)
        plt.title("Classified Surface Map\nProjection: UTM Zone 23", fontsize = 30), ax1.set_aspect('equal')
        plt.xticks([0, 2745, 5490],['-51.000235','-49.708602','-48.418656'],fontsize=26, rotation=45),plt.xlabel('Longitude (decimal degrees)',fontsize=26)
        plt.yticks([0,2745, 5490],['67.615437','67.610307','67.594927'],fontsize=26),plt.ylabel('Latitude (decimal degrees)',fontsize=26)
        plt.grid(None)

        # second subplot = albedo map
        ax2 = plt.subplot(212)
        img2 = plt.imshow(albedo, cmap=cmap2, vmin=0, vmax=1)
        cbar2 = plt.colorbar(mappable=img2, fraction=0.045)
        cbar2.ax.set_yticklabels(labels=[0,0.2,0.4,0.6,0.8,1.0], fontsize=26)
        cbar2.set_label('Albedo',fontsize=26)
        plt.xticks([0, 2745, 5490],['-51.000235','-49.708602','-48.418656'],fontsize=26,rotation=45),plt.xlabel('Longitude (decimal degrees)',fontsize=26)
        plt.yticks([0,2745, 5490],['67.615437','67.610307','67.594927'],fontsize=26),plt.ylabel('Latitude (decimal degrees)',fontsize=26)
        plt.grid(None),plt.title("Albedo Map\nProjection: UTM Zone 23",fontsize=30)
        ax2.set_aspect('equal')
        plt.tight_layout()

        plt.savefig(str(savefig_path + "Sentinel_Classified_Albedo.png"), dpi=100)

        if not savefigs:
            plt.show()

    if savefigs:
        plt.savefig(str(savefig_path + "Sentinel_Classified_Albedo.png"), dpi=300)

    return predicted, albedo, dataset

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


"""
*** INFO ***

code written by Joseph Cook (University of Sheffield), 2018. Correspondence to joe.cook@sheffield.ac.uk

*** OVERVIEW ***

This code trains a random forest classifier on ground-level reflectance data, then deploys it to classify Sentinel 2
imagery into various surface categories:

Water, Snow, Clean Ice, Cryoconite, Light Algal Bloom, Heavy Algal Bloom

The result is a classified ice surface map. The coverage statistics are calculated and reported.
The albedo of each pixel is calculated from the multispectral reflectance using Liang et al's (2002) narrowband to
broadband conversion formula. The result is an albedo map of the ice surface.

Both the classified map and albedo map are trimmed to remove non-ice areas using the Greenland Ice Mapping Project mask
before spatial stats are calculated.


*** PREREQUISITES ***

1) The core training data is saved as a csv named "HRCF_master_machine_snicar.csv". This is a collection of spectral
reflectance measurements made at ground level using an ASD Field Spec Pro 3 for surfaces whose characteristics are
known.

2) Sentinel-2 band images. Folders containing Level 1C products can be downloaded from Earthexplorer.usgs.gov
These must be converted from level 1C to Level 2A (i.e. corrected for atmospheric effects and reprojected to a
consistent 20m ground resolution) using the ESA command line tool Sen2Cor.

This requires downloading the Sen2Cor software and running from the command line. Instructions are available here:
https://forum.step.esa.int/t/sen2cor-2-4-0-stand-alone-installers-how-to-install/6908

Sen2Cor details:

L2A processor path =  '/home/joe/Sen2Cor/Sen2Cor-02.05.05-Linux64/bin/L2A_Process'
Default configuration file = '/home/joe/sen2cor/2.5/cfg/L2A_GIPP.xml'

With file downloaded from EarthExplorer on desktop, L1C to L2A processing achieved using command:

>> /home/joe/Sen2Cor/Sen2Cor-02.05.05-Linux64/bin/L2A_Process ...
>> /home/joe/Desktop/S2A_MSIL1C_20160721T151912_N0204_R068_T22WEV_20160721T151913.SAFE

Then reformat the jp2 images for each band into netCDF files using gdal:

>> source activate IceSurfClassifiers
>> cd /home/joe/Desktop/S2A_MSIL2A_20160721T151912_N0204_R068_T22WEV_20160721T151913.SAFE/GRANULE/L2A_T22WEV_A005642_
   20160721T151913/IMG_DATA/R20m/

>> gdal_translate L2A_T22WEV_20160721T151912_B02_20m.jp2 /home/joe/Desktop/S2A_NetCDFs/SITENAME/B02.nc

repeat for each band. The final processed files are then available in the desktop folder 'S2A_NetCDFs' and saved under a
site name, then as B02.nc, B03.nc etc. These netcdfs are then used as input data in this script.

The resulting NetCDF files are then used as input data for this script.

3) The GIMP mask downloaded from https://nsidc.org/data/nsidc-0714/versions/1 must be saved to the working directory.
Ensure the downloaded tile is the correct one for the section of ice sheet being examined.


*** FUNCTIONS***

This script is divided into several functions. The first function (create_dataset) preprocesses the raw data into a
format appropriate for supervised classification (i.e. explicit features and labels). The raw hyperspectral data is
first organised into separate pandas dataframes for each surface class. The data is then reduced down to the reflectance
at the nine key wavelengths coincident with those of the Sentinel 2 spectrometer. The remaining data are discarded.
The dataset is then arranged into columns with one column per wavelength and a separate column for the surface label.
The dataset's features are the reflectance at each wavelength, and the labels are the surface types.
The dataframes for each surface type are merged into one large dataframe and then the labels are removed and saved as a
separate dataframe. No scaling of the data is required because the reflectance is already normalised between 0 and 1 by
the spectrometer.

The second function (train_test_split) separates the dataset into training and test sets and trains a random forest
classifier. Setting n_jobs = -1 ensures the training and prediction phases are distributed over all available processor
cores. The performance of the trained model is calculated and displayed. The classifier can optionally be saved to a
.pkl file, or loaded in from an external .pkl file rather than continually retraining on the fly.

The third function (format_mask) reprojects the GIMP mask to an identical coordinate system, pixel size and spatial
extent to the Sentinel 2 images and returns a Boolean numpy array that will later be used to mask out non-ice areas
of the classificed map and albedo map

The fourth function applies the trained classifier to the sentinel 2 images and masks out non-ice areas, then applies
Liang et al(2002) narrowband to broadband conversion formula, producing a NetCDF file containing all the data arrays and
metadata along with a plot of each map.startTime1 = datetime.now()

The final function calculates spatial statistics for the classified surface and albed maps.

"""


###########################################################################################
############################# IMPORT MODULES #########################################

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import xarray as xr
import seaborn as sn
from osgeo import gdal, osr
import georaster

# matplotlib settings: use ggplot style and turn interactive mode off so that plots can be saved and not shown (for
# rapidly processing multiple images later)

mpl.style.use('ggplot')
plt.ioff()

# DEFINE FUNCTIONS
def set_paths(virtual_machine = False):

    if not virtual_machine:
        savefig_path = '/home/joe/Code/IceSurfClassifiers/Sentinel_Outputs/'
        img_path = '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2A_NetCDFs/KGR/'
        HCRF_file = '/home/joe/Code/IceSurfClassifiers/Training_Data/HCRF_master_machine_snicar.csv'
        # paths for format_mask()
        Sentinel_template = '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/S2_L2A_KGR/GRANULE/L2A_T22WEV_A005642_20160721T151913/IMG_DATA/R20m/L2A_T22WEV_20160721T151912_B02_20m.jp2'
        mask_in = '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/Mask/GIMP_MASK.tif'
        mask_out = '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/Mask/GIMP_MASK.nc'

    else:
        # Virtual Machine
        # paths for create_dataset()
        savefig_path = '/data/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_Resources/Sentinel_Outputs/'
        img_path = '/data/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_Resources/S2A_NetCDFs/'
        HCRF_file = '/data/home/tothepoles/PycharmProjects/IceSurfClassifiers/Training_Data/HCRF_master_machine_snicar.csv'
        # paths for format_mask()
        Sentinel_template = '/data/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_Resources/L2A_T22WEV_20160721T151912_B02_20m.jp2'
        mask_path = ['/data/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_Resources/']
        mask_in = '/data/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_Resources/GIMP_MASK.tif'
        mask_out = '/data/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_Resources/GIMP_MASK.nc'

    return HCRF_file, savefig_path,img_path, Sentinel_template, mask_in, mask_out


def create_dataset(HCRF_file, img_path, plot_spectra=True, savefigs=True):
# Sentinel 2 dataset
# create 3D numpy array with dim1 = band, dims 2 and 3 = spatial x and y. Values are reflectance.
    S2vals = np.zeros([9,5490,5490])
    bands = ['02','03','04','05','05','07','08','11','12']
    for i in np.arange(0,len(bands),1):
        S2BX = xr.open_dataset(str(img_path+'B'+bands[i]+'.nc'))
        S2BXarr = S2BX.to_array()
        S2BXvals = np.array(S2BXarr.variable.values[1])
        S2BXvals = S2BXvals.astype(float)
        S2vals[i,:,:] = S2BXvals

    S2vals = S2vals/10000 # correct unit from S2 L2A data to reflectance between 0-1

# ground reflectance dataset
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


    Zero = pd.DataFrame()
    Zero['R140'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R210'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R315'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R355'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R390'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R433'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R515'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R1260'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R1840'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['label'] = 'Zero'


    X = X.append(Y, ignore_index=True)
    X = X.append(Z, ignore_index=True)
    X = X.append(P, ignore_index=True)
    X = X.append(Q, ignore_index=True)
    X = X.append(R, ignore_index=True)
    X = X.append(Zero, ignore_index=True)

    return S2vals, X

def split_train_test(X, test_size=0.2, n_trees= 64, print_conf_mx = True, plot_conf_mx = True, savefigs = False,
                     show_model_performance = True, pickle_model=False):

    XX = X.drop(['label'],axis=1)
    YY = X['label']
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(XX, YY, test_size=test_size)
    clf = RandomForestClassifier(n_estimators=n_trees, max_leaf_nodes=16, n_jobs=-1)
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
                   xticklabels=['NaN', 'Snow', 'Water', 'Cryoconite','Clean Ice', 'Light Algae', 'Heavy Algae'],
                   yticklabels=['NaN', 'Snow', 'Water', 'Cryoconite', 'Clean Ice', 'Light Algae', 'Heavy Algae'],
                   cbar_kws={"shrink": 0.4, 'label':'frequency'}, ax=ax1), ax1.tick_params(axis='both', rotation=45)
        ax1.set_title('Confusion Matrix'), ax1.set_aspect('equal')

        sn.heatmap(norm_conf_mx, annot=True, annot_kws={"size": 16}, cmap=plt.cm.gray,
                   xticklabels=['NaN', 'Snow', 'Water', 'Cryoconite', 'Clean Ice', 'Light Algae', 'Heavy Algae'],
                   yticklabels=['NaN', 'Snow', 'Water', 'Cryoconite', 'Clean Ice', 'Light Algae', 'Heavy Algae'],
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


def format_mask (Sentinel_template,mask_in,mask_out):
    """
    Function reprojects GIMP mask to dimensions, resolution and spatial coords of the S2 images, enabling
    Boolean masking of land-ice area.

    INPUTS:
    Sentinel = Sentinel image to use as projection template
    mask_in = file path to mask file
    mask_out = file path to save reprojected mask
    res = resolution to save mask

    OUTPUTS:
    reprojected mask in .tif format

    """
    mask = gdal.Open(mask_in)

    mask_proj = mask.GetProjection()
    mask_geotrans = mask.GetGeoTransform()
    data_type = mask.GetRasterBand(1).DataType
    n_bands = mask.RasterCount

    Sentinel = gdal.Open(Sentinel_template)

    Sentinel_proj = Sentinel.GetProjection()
    Sentinel_geotrans = Sentinel.GetGeoTransform()
    w = Sentinel.RasterXSize
    h = Sentinel.RasterYSize

    mask_filename = mask_out
    new_mask = gdal.GetDriverByName('NETCDF').Create(mask_filename,
                                                w, h, n_bands, data_type)
    new_mask.SetGeoTransform(Sentinel_geotrans)
    new_mask.SetProjection(Sentinel_proj)

    gdal.ReprojectImage( mask, new_mask, mask_proj,
                         Sentinel_proj, gdal.GRA_NearestNeighbour)
    new_mask = None  # Flush disk

    # open netCDF mask and extract values to numpy array
    maskxr = xr.open_dataset(mask_out)
    mask_array = np.array(maskxr.Band1.values)
    #replace nans with 0 to create binary numerical array
    nans = np.isnan(mask_array)
    mask_array[nans]=0

    return mask_array


def ClassifyImages(S2vals, clf, mask_array, plot_maps = True, savefigs=False, save_netcdf = False):

    # get dimensions of each band layer
    lenx, leny = np.shape(S2vals[0])

    # convert image bands into single 5-dimensional numpy array
    S2valsT = S2vals.reshape(9, lenx * leny)  # reshape into 5 x 1D arrays
    S2valsT = S2valsT.transpose()  # transpose so that bands are read as features


    # create albedo array by applying Knap (1999) narrowband - broadband conversion
    albedo_array = np.array([0.356 * (S2vals[0]) + 0.13 * (S2vals[2]) + 0.373 * (
            S2vals[6]) + 0.085 * (S2vals[7]) + 0.072 * (S2vals[8]) - 0.0018])

    # apply ML algorithm to 4-value array for each pixel - predict surface type
    predicted = clf.predict(S2valsT)
    predicted = np.array(predicted)

    # convert surface class (string) to a numeric value for plotting
    predicted[predicted == 'SN'] = float(1)
    predicted[predicted == 'WAT'] = float(2)
    predicted[predicted == 'CC'] = float(3)
    predicted[predicted == 'CI'] = float(4)
    predicted[predicted == 'LA'] = float(5)
    predicted[predicted == 'HA'] = float(6)
    predicted[predicted == 'Zero'] = float(0)

    # ensure array data type is float (required for imshow)
    predicted = predicted.astype(float)

    # reshape 1D array back into original image dimensions

    predicted = np.reshape(predicted, [lenx, leny])
    albedo = np.reshape(albedo_array, [lenx, leny])


    # apply GIMP mask to ignore non-ice surfaces
    predicted = np.ma.masked_where(mask_array==0, predicted)
    albedo = np.ma.masked_where(mask_array==0, albedo)

    # mask out areas not covered by Sentinel tile, but not excluded by GIMP mask
    predicted = np.ma.masked_where(predicted <=0, predicted)
    albedo = np.ma.masked_where(albedo <=0, albedo)

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

    # 2) Create associated lat/lon coordinates DataArrays using georaster (imports geo metadata without loading img)
    # see georaster docs at https: // media.readthedocs.org / pdf / georaster / latest / georaster.pdf
    S2 = georaster.SingleBandRaster('NETCDF:"%s":Band1' % (str(img_path+'B02.nc')),
                                    load_data=False)
    lon, lat = S2.coordinates(latlon=True)
    S2 = None  # close file
    S2 = xr.open_dataset((str(img_path+'B02.nc')), chunks={'x': 2000, 'y': 2000})
    coords_geo = {'y': S2['y'], 'x': S2['x']}
    S2 = None  # close file

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
    predictedxr = xr.DataArray(predicted, coords=coords_geo, dims=['y', 'x'])
    predictedxr.encoding = {'dtype': 'int16', 'zlib': True, '_FillValue': -9999}
    predictedxr.name = 'Surface Class'
    predictedxr.attrs['long_name'] = 'Surface classified using Random Forest'
    predictedxr.attrs['units'] = 'None'
    predictedxr.attrs[
        'key'] = 'Unknown:0; Snow:1; Water:2; Cryoconite:3; Clean Ice:4; Light Algae:5; Heavy Algae:6'
    predictedxr.attrs['grid_mapping'] = 'UTM'

    # add albedo map array and add metadata
    albedoxr = xr.DataArray(albedo, coords=coords_geo, dims=['y', 'x'])
    albedoxr.encoding = {'dtype': 'int16', 'scale_factor': 0, 'zlib': True, '_FillValue': -9999}
    albedoxr.name = 'Surface albedo computed after Knap et al. (1999) narrowband-to-broadband conversion'
    albedoxr.attrs['units'] = 'dimensionless'
    albedoxr.attrs['grid_mapping'] = 'UTM'

    # collate data arrays into a dataset
    dataset = xr.Dataset({

        'classified': (['x', 'y'], predictedxr),
        'albedo': (['x', 'y'], albedoxr),
        'mask': (['x','y'],mask_array),
        'Projection': proj_info,
        'longitude': (['x', 'y'], lon_array),
        'latitude': (['x', 'y'], lat_array)
    })

    # add metadata for dataset
    dataset.attrs['Conventions'] = 'CF-1.4'
    dataset.attrs['Author'] = 'Joseph Cook (University of Sheffield, UK)'
    dataset.attrs[
        'title'] = 'Classified surface and albedo maps produced from Sentinel-2 ' \
                   'imagery of the SW Greenland Ice Sheet'

    # Additional geo-referencing
    dataset.attrs['nx'] = len(dataset.x)
    dataset.attrs['ny'] = len(dataset.y)
    dataset.attrs['xmin'] = float(dataset.x.min())
    dataset.attrs['ymax'] = float(dataset.y.max())
    dataset.attrs['spacing'] = 20

    # NC conventions metadata for dimensions variables
    dataset.x.attrs['units'] = 'meters'
    dataset.x.attrs['standard_name'] = 'projection_x_coordinate'
    dataset.x.attrs['point_spacing'] = 'even'
    dataset.x.attrs['axis'] = 'x'

    dataset.y.attrs['units'] = 'meters'
    dataset.y.attrs['standard_name'] = 'projection_y_coordinate'
    dataset.y.attrs['point_spacing'] = 'even'
    dataset.y.attrs['axis'] = 'y'

    # save dataset to netcdf if requested
    if save_netcdf:
        dataset.to_netcdf(savefig_path + "Classification_and_Albedo_Data.nc")

    if plot_maps or savefigs:

        cmap1 = mpl.colors.ListedColormap(
            ['purple', 'white', 'royalblue', 'black', 'lightskyblue', 'mediumseagreen', 'darkgreen'])
        cmap1.set_under(color='white')  # make sure background is white
        cmap2 = plt.get_cmap('Greys_r')  # reverse greyscale for albedo
        cmap2.set_under(color='white')  # make sure background is white

        fig = plt.figure(figsize=(15, 30))

        # first subplot = classified map
        class_labels = ['Unknown', 'Snow', 'Water', 'Cryoconite', 'Clean Ice', 'Light Algae', 'Heavy Algae']
        ax1 = plt.subplot(211)
        img = plt.imshow(predicted, cmap=cmap1,vmin=0, vmax=7)
        cbar = fig.colorbar(mappable = img, ax=ax1, fraction=0.045)

        n_classes = len(class_labels)
        tick_locs = np.arange(0.5,len(class_labels),1)
        cbar.set_ticks(tick_locs)
        cbar.ax.set_yticklabels(class_labels, fontsize=26, rotation=0, va='center')
        cbar.set_label('Surface Class',fontsize=22)
        plt.title("Classified Surface Map\nProjection: UTM Zone 23", fontsize = 30), ax1.set_aspect('equal')
        plt.xticks([0, 2745, 5490],['-51.000235','-49.708602','-48.418656'],fontsize=26, rotation=45),plt.xlabel('Longitude (decimal degrees)',fontsize=26)
        plt.yticks([0,2745, 5490],['67.615437','67.610307','67.594927'],fontsize=26),plt.ylabel('Latitude (decimal degrees)',fontsize=26)
        plt.grid(None)

        # second subplot = albedo map
        ax2 = plt.subplot(212)
        img2 = plt.imshow(albedo, cmap=cmap2, vmin=0, vmax=1)
        cbar2 = plt.colorbar(mappable=img2, fraction=0.045)
        cbar2.ax.set_yticklabels(labels=[0,0.2,0.4,0.6,0.8,1.0], fontsize=26)
        cbar2.set_label('Albedo',fontsize=26)
        plt.xticks([0, 2745, 5490],['-51.000235','-49.708602','-48.418656'],fontsize=26,rotation=45),plt.xlabel('Longitude (decimal degrees)',fontsize=26)
        plt.yticks([0,2745, 5490],['67.615437','67.610307','67.594927'],fontsize=26),plt.ylabel('Latitude (decimal degrees)',fontsize=26)
        plt.grid(None),plt.title("Albedo Map\nProjection: UTM Zone 23",fontsize=30)
        ax2.set_aspect('equal')
        plt.tight_layout()

        plt.savefig(str(savefig_path + "Sentinel_Classified_Albedo.png"), dpi=100)

        if not savefigs:
            plt.show()

    if savefigs:
        plt.savefig(str(savefig_path + "Sentinel_Classified_Albedo.png"), dpi=300)

    return predicted, albedo, dataset

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



# RUN AND TIME FUNCTIONS
HCRF_file, savefig_path,img_path, Sentinel_template, mask_in, mask_out = set_paths(virtual_machine=False)

#create dataset
S2vals, X = create_dataset(HCRF_file, img_path, plot_spectra=False, savefigs=False)

#format mask
mask_array = format_mask (Sentinel_template,mask_in,mask_out)

#optimise and train model
clf, conf_mx_RF, norm_conf_mx = split_train_test(X, test_size=0.3, n_trees= 32, print_conf_mx = True, plot_conf_mx = True, savefigs = True,
                     show_model_performance = True, pickle_model=False)

# apply model to Sentinel2 image
predicted, albedo, dataset =  ClassifyImages(S2vals,clf, mask_array, plot_maps = False, savefigs=True, save_netcdf=True)


# calculate spatial stats
albedoDF = albedo_report(predicted, albedo, save_albedo_data = False)

