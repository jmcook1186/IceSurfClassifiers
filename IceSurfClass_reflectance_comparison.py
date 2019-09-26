import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from osgeo import gdal

plt.style.use('classic')
###### UAV #######
# open UAV datasets

def UAV():
    with xr.open_dataset('/home/joe/Code/IceSurfClassifiers/UAV_Resources/uav_data.nc').transpose('y','x') as uav:
        with xr.open_dataset('/home/joe/Code/IceSurfClassifiers//UAV_Outputs/Classification_and_Albedo_Data_UAV.nc') as uav_proc:


            # set band indices and concatenate into a 5 x 6480 x 6600 dataset
            band_idx = pd.Index([1, 2, 3, 4, 5], name='bands')
            uav['Band1'] -= 0.17
            uav['Band2'] -= 0.18
            uav['Band3'] -= 0.15
            uav['Band4'] -= 0.16
            uav['Band5'] -= 0.05
            concat = xr.concat([uav.Band1, uav.Band2, uav.Band3, uav.Band4, uav.Band5], band_idx)

            # Mask nodata areas
            concat2 = concat.where(concat.sum(dim='bands') > 0)
            concat2 = concat2.rename({"x":"y","y":"x"}) # flip coords to match processed uav

            # key = 'Snow:1; Water:2; Cryoconite:3; Clean Ice:4; Light Algae:5; Heavy Algae:6'

            # separate into classes, each with reflectance values in all 5 bands
            HA = concat2.where(uav_proc.classified.values>=6,np.nan)
            LA = concat2.where(uav_proc.classified.values==5,np.nan)
            CI = concat2.where(uav_proc.classified.values==4,np.nan)
            CC = concat2.where(uav_proc.classified.values==3,np.nan)
            WAT = concat2.where(uav_proc.classified.values==2,np.nan)
            SN = concat2.where(uav_proc.classified.values<=1,np.nan)

            # organise mean, max and min reflectance into dataframe
            UAV_HA = pd.DataFrame(columns = ['B1','B2','B3','B4','B5'], index=['mean','max','min','std'])
            UAV_HA['B1'] = [HA[0,:,:].mean().values,HA[0,:,:].max().values, HA[0,:,:].min().values,HA[0,:,:].std().values]
            UAV_HA['B2'] = [HA[1,:,:].mean().values,HA[1,:,:].max().values, HA[1,:,:].min().values,HA[1,:,:].std().values]
            UAV_HA['B3'] = [HA[2,:,:].mean().values,HA[2,:,:].max().values, HA[2,:,:].min().values,HA[2,:,:].std().values]
            UAV_HA['B4'] = [HA[3,:,:].mean().values,HA[3,:,:].max().values, HA[3,:,:].min().values,HA[3,:,:].std().values]
            UAV_HA['B5'] = [HA[4,:,:].mean().values,HA[4,:,:].max().values, HA[4,:,:].min().values,HA[4,:,:].std().values]

            UAV_LA = pd.DataFrame(columns = ['B1','B2','B3','B4','B5'], index=['mean','max','min','std'])
            UAV_LA['B1'] = [LA[0,:,:].mean().values,LA[0,:,:].max().values, LA[0,:,:].min().values,LA[0,:,:].std().values]
            UAV_LA['B2'] = [LA[1,:,:].mean().values,LA[1,:,:].max().values, LA[1,:,:].min().values,LA[1,:,:].std().values]
            UAV_LA['B3'] = [LA[2,:,:].mean().values,LA[2,:,:].max().values, LA[2,:,:].min().values,LA[2,:,:].std().values]
            UAV_LA['B4'] = [LA[3,:,:].mean().values,LA[3,:,:].max().values, LA[3,:,:].min().values,LA[3,:,:].std().values]
            UAV_LA['B5'] = [LA[4,:,:].mean().values,LA[4,:,:].max().values, LA[4,:,:].min().values,LA[4,:,:].std().values]

            UAV_CI = pd.DataFrame(columns = ['B1','B2','B3','B4','B5'], index=['mean','max','min','std'])
            UAV_CI['B1'] = [CI[0,:,:].mean().values,CI[0,:,:].max().values, CI[0,:,:].min().values,CI[0,:,:].std().values]
            UAV_CI['B2'] = [CI[1,:,:].mean().values,CI[1,:,:].max().values, CI[1,:,:].min().values,CI[1,:,:].std().values]
            UAV_CI['B3'] = [CI[2,:,:].mean().values,CI[2,:,:].max().values, CI[2,:,:].min().values,CI[2,:,:].std().values]
            UAV_CI['B4'] = [CI[3,:,:].mean().values,CI[3,:,:].max().values, CI[3,:,:].min().values,CI[3,:,:].std().values]
            UAV_CI['B5'] = [CI[4,:,:].mean().values,CI[4,:,:].max().values, CI[4,:,:].min().values,CI[4,:,:].std().values]

            UAV_CC = pd.DataFrame(columns = ['B1','B2','B3','B4','B5'], index=['mean','max','min','std'])
            UAV_CC['B1'] = [CC[0,:,:].mean().values,CC[0,:,:].max().values, CC[0,:,:].min().values,CC[0,:,:].std().values]
            UAV_CC['B2'] = [CC[1,:,:].mean().values,CC[1,:,:].max().values, CC[1,:,:].min().values,CC[1,:,:].std().values]
            UAV_CC['B3'] = [CC[2,:,:].mean().values,CC[2,:,:].max().values, CC[2,:,:].min().values,CC[2,:,:].std().values]
            UAV_CC['B4'] = [CC[3,:,:].mean().values,CC[3,:,:].max().values, CC[3,:,:].min().values,CC[3,:,:].std().values]
            UAV_CC['B5'] = [CC[4,:,:].mean().values,CC[4,:,:].max().values, CC[4,:,:].min().values,CC[4,:,:].std().values]

            UAV_WAT = pd.DataFrame(columns = ['B1','B2','B3','B4','B5'], index=['mean','max','min','std'])
            UAV_WAT['B1'] = [WAT[0,:,:].mean().values,WAT[0,:,:].max().values, WAT[0,:,:].min().values,WAT[0,:,:].std().values]
            UAV_WAT['B2'] = [WAT[1,:,:].mean().values,WAT[1,:,:].max().values, WAT[1,:,:].min().values,WAT[1,:,:].std().values]
            UAV_WAT['B3'] = [WAT[2,:,:].mean().values,WAT[2,:,:].max().values, WAT[2,:,:].min().values,WAT[2,:,:].std().values]
            UAV_WAT['B4'] = [WAT[3,:,:].mean().values,WAT[3,:,:].max().values, WAT[3,:,:].min().values,WAT[3,:,:].std().values]
            UAV_WAT['B5'] = [WAT[4,:,:].mean().values,WAT[4,:,:].max().values, WAT[4,:,:].min().values,WAT[4,:,:].std().values]

            UAV_SN = pd.DataFrame(columns = ['B1','B2','B3','B4','B5'], index=['mean','max','min','std'])
            UAV_SN['B1'] = [SN[0,:,:].mean().values,SN[0,:,:].max().values, SN[0,:,:].min().values,SN[0,:,:].std().values]
            UAV_SN['B2'] = [SN[1,:,:].mean().values,SN[1,:,:].max().values, SN[1,:,:].min().values,SN[1,:,:].std().values]
            UAV_SN['B3'] = [SN[2,:,:].mean().values,SN[2,:,:].max().values, SN[2,:,:].min().values,SN[2,:,:].std().values]
            UAV_SN['B4'] = [SN[3,:,:].mean().values,SN[3,:,:].max().values, SN[3,:,:].min().values,SN[3,:,:].std().values]
            UAV_SN['B5'] = [SN[4,:,:].mean().values,SN[4,:,:].max().values, SN[4,:,:].min().values,SN[4,:,:].std().values]

            # plot data

    return UAV_CI, UAV_CC, UAV_HA, UAV_LA, UAV_SN, UAV_WAT


#### S2 ####
############
def S2():

    img_path = '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2017/S2A_MSIL2A_20170726T151911_N0205_R068_T22WEV_20170726T151917.SAFE/GRANULE/L2A_T22WEV_A010933_20170726T151917/IMG_DATA/R20m/L2A_T22WEV_20170726T151911_'

    daB2 = xr.open_rasterio(str(img_path +'B02_20m.jp2'), chunks={'x': 2000, 'y': 2000})
    daB3 = xr.open_rasterio(str(img_path +'B03_20m.jp2'), chunks={'x': 2000, 'y': 2000})
    daB4 = xr.open_rasterio(str(img_path +'B04_20m.jp2'), chunks={'x': 2000, 'y': 2000})
    daB5 = xr.open_rasterio(str(img_path +'B05_20m.jp2'), chunks={'x': 2000, 'y': 2000})
    daB6 = xr.open_rasterio(str(img_path +'B06_20m.jp2'), chunks={'x': 2000, 'y': 2000})
    daB7 = xr.open_rasterio(str(img_path +'B07_20m.jp2'), chunks={'x': 2000, 'y': 2000})
    daB8 = xr.open_rasterio(str(img_path +'B8A_20m.jp2'), chunks={'x': 2000, 'y': 2000})
    daB11 = xr.open_rasterio(str(img_path +'B11_20m.jp2'), chunks={'x': 2000, 'y': 2000})
    daB12 = xr.open_rasterio(str(img_path +'B12_20m.jp2'), chunks={'x': 2000, 'y': 2000})

    daB2 = xr.DataArray.squeeze(daB2, dim='band')
    daB3 = xr.DataArray.squeeze(daB3, dim='band')
    daB4 = xr.DataArray.squeeze(daB4, dim='band')
    daB5 = xr.DataArray.squeeze(daB5, dim='band')
    daB6 = xr.DataArray.squeeze(daB6, dim='band')
    daB7 = xr.DataArray.squeeze(daB7, dim='band')
    daB8 = xr.DataArray.squeeze(daB8, dim='band')
    daB11 = xr.DataArray.squeeze(daB11, dim='band')
    daB12 = xr.DataArray.squeeze(daB12, dim='band')

    S2vals = xr.Dataset({'B02': (('x', 'y'), daB2.values / 10000), 'B03': (('x', 'y'), daB3.values / 10000),
                         'B04': (('y', 'x'), daB4.values / 10000), 'B05': (('y', 'x'), daB5.values / 10000),
                         'B06': (('y', 'x'), daB6.values / 10000),
                         'B07': (('y', 'x'), daB7.values / 10000), 'B08': (('y', 'x'), daB8.values / 10000),
                         'B11': (('y', 'x'), daB11.values / 10000),
                         'B12': (('y', 'x'), daB12.values / 10000)})


    with xr.open_dataset('/home/joe/Code/IceSurfClassifiers/Sentinel_Outputs/2017_Outputs/T22WEV_Classification_and_Albedo_Data_2017.nc') as S2proc:

        # divide into bands
        # key = 'Snow:1; Water:2; Cryoconite:3; Clean Ice:4; Light Algae:5; Heavy Algae:6'
        HAB2 = S2vals.B02.where(S2proc.classified.values>=6,0)
        HAB2=HAB2.values[HAB2.values>0]
        HAB3 = S2vals.B03.where(S2proc.classified.values>=6,0)
        HAB3=HAB3.values[HAB3.values>0]
        HAB4 = S2vals.B04.where(S2proc.classified.values>=6,0)
        HAB4=HAB4.values[HAB4.values>0]
        HAB5 = S2vals.B05.where(S2proc.classified.values>=6,0)
        HAB5=HAB5.values[HAB5.values>0]
        HAB6 = S2vals.B06.where(S2proc.classified.values>=6,0)
        HAB6=HAB6.values[HAB6.values>0]
        HAB7 = S2vals.B07.where(S2proc.classified.values>=6,0)
        HAB7=HAB7.values[HAB7.values>0]
        HAB8 = S2vals.B08.where(S2proc.classified.values>=6,0)
        HAB8=HAB8.values[HAB8.values>0]
        HAB11 = S2vals.B11.where(S2proc.classified.values>=6,0)
        HAB11=HAB11.values[HAB11.values>0]
        HAB12 = S2vals.B12.where(S2proc.classified.values>=6,0)
        HAB12=HAB12.values[HAB12.values>0]

        LAB2 = S2vals.B02.where(S2proc.classified.values==5).values
        LAB2=LAB2[LAB2>0]
        LAB3 = S2vals.B03.where(S2proc.classified.values==5).values
        LAB3=LAB3[LAB3>0]
        LAB4 = S2vals.B04.where(S2proc.classified.values==5).values
        LAB4=LAB4[LAB4>0]
        LAB5 = S2vals.B05.where(S2proc.classified.values==5).values
        LAB5=LAB5[LAB5>0]
        LAB6 = S2vals.B06.where(S2proc.classified.values==5).values
        LAB6=LAB6[LAB6>0]
        LAB7 = S2vals.B07.where(S2proc.classified.values==5).values
        LAB7=LAB7[LAB7>0]
        LAB8 = S2vals.B08.where(S2proc.classified.values==5).values
        LAB8=LAB8[LAB8>0]
        LAB11 = S2vals.B11.where(S2proc.classified.values==5).values
        LAB11=LAB11[LAB11>0]
        LAB12 = S2vals.B12.where(S2proc.classified.values==5).values
        LAB12=LAB12[LAB12>0]

        CIB2 = S2vals.B02.where(S2proc.classified.values==4).values
        CIB2=CIB2[CIB2>0]
        CIB3 = S2vals.B03.where(S2proc.classified.values==4).values
        CIB3=CIB3[CIB3>0]
        CIB4 = S2vals.B04.where(S2proc.classified.values==4).values
        CIB4=CIB4[CIB4>0]
        CIB5 = S2vals.B05.where(S2proc.classified.values==4).values
        CIB5=CIB5[CIB5>0]
        CIB6 = S2vals.B06.where(S2proc.classified.values==4).values
        CIB6=CIB6[CIB6>0]
        CIB7 = S2vals.B07.where(S2proc.classified.values==4).values
        CIB7=CIB7[CIB7>0]
        CIB8 = S2vals.B08.where(S2proc.classified.values==4).values
        CIB8=CIB8[CIB8>0]
        CIB11 = S2vals.B11.where(S2proc.classified.values==4).values
        CIB11=CIB11[CIB11>0]
        CIB12 = S2vals.B12.where(S2proc.classified.values==4).values
        CIB12=CIB12[CIB12>0]

        CCB2 = S2vals.B02.where(S2proc.classified.values==3).values
        CCB2=CCB2[CCB2>0]
        CCB3 = S2vals.B03.where(S2proc.classified.values==3).values
        CCB3=CCB3[CCB3>0]
        CCB4 = S2vals.B04.where(S2proc.classified.values==3).values
        CCB4=CCB4[CCB4>0]
        CCB5 = S2vals.B05.where(S2proc.classified.values==3).values
        CCB5=CCB5[CCB5>0]
        CCB6 = S2vals.B06.where(S2proc.classified.values==3).values
        CCB6=CCB6[CCB6>0]
        CCB7 = S2vals.B07.where(S2proc.classified.values==3).values
        CCB7=CCB7[CCB7>0]
        CCB8 = S2vals.B08.where(S2proc.classified.values==3).values
        CCB8=CCB8[CCB8>0]
        CCB11 = S2vals.B11.where(S2proc.classified.values==3).values
        CCB11=CCB11[CCB11>0]
        CCB12 = S2vals.B12.where(S2proc.classified.values==3).values
        CCB12=CCB12[CCB12>0]

        WATB2 = S2vals.B02.where(S2proc.classified.values==2).values
        WATB2=WATB2[WATB2>0]
        WATB3 = S2vals.B03.where(S2proc.classified.values==2).values
        WATB3=WATB3[WATB3>0]
        WATB4 = S2vals.B04.where(S2proc.classified.values==2).values
        WATB4=WATB4[WATB4>0]
        WATB5 = S2vals.B05.where(S2proc.classified.values==2).values
        WATB5=WATB5[WATB5>0]
        WATB6 = S2vals.B06.where(S2proc.classified.values==2).values
        WATB6=WATB6[WATB6>0]
        WATB7 = S2vals.B07.where(S2proc.classified.values==2).values
        WATB7=WATB7[WATB7>0]
        WATB8 = S2vals.B08.where(S2proc.classified.values==2).values
        WATB8=WATB8[WATB8>0]
        WATB11 = S2vals.B11.where(S2proc.classified.values==2).values
        WATB11=WATB11[WATB11>0]
        WATB12 = S2vals.B12.where(S2proc.classified.values==2).values
        WATB12=WATB12[WATB12>0]

        SNB2 = S2vals.B02.where(S2proc.classified.values==1).values
        SNB2=SNB2[SNB2>0]
        SNB3 = S2vals.B03.where(S2proc.classified.values==1).values
        SNB3=SNB3[SNB3>0]
        SNB4 = S2vals.B04.where(S2proc.classified.values==1).values
        SNB4=SNB4[SNB4>0]
        SNB5 = S2vals.B05.where(S2proc.classified.values==1).values
        SNB5=SNB5[SNB5>0]
        SNB6 = S2vals.B06.where(S2proc.classified.values==1).values
        SNB6=SNB6[SNB6>0]
        SNB7 = S2vals.B07.where(S2proc.classified.values==1).values
        SNB7=SNB7[SNB7>0]
        SNB8 = S2vals.B08.where(S2proc.classified.values==1).values
        SNB8=SNB8[SNB8>0]
        SNB11 = S2vals.B11.where(S2proc.classified.values==1).values
        SNB11=SNB11[SNB11>0]
        SNB12 = S2vals.B12.where(S2proc.classified.values==1).values
        SNB12=SNB12[SNB12>0]

    S2_SN = pd.DataFrame(columns=['B2','B3','B4','B5','B6','B7','B8','B11','B12'], index=['mean', 'max', 'min', 'std'])
    S2_SN['B2'] = [SNB2.mean(), SNB2.max(), SNB2.min(),SNB2.std()]
    S2_SN['B3'] = [SNB3.mean(), SNB3.max(), SNB3.min(), SNB3.std()]
    S2_SN['B4'] = [SNB4.mean(), SNB4.max(), SNB4.min(), SNB4.std()]
    S2_SN['B5'] = [SNB5.mean(), SNB5.max(), SNB5.min(), SNB5.std()]
    S2_SN['B6'] = [SNB6.mean(), SNB6.max(), SNB6.min(), SNB6.std()]
    S2_SN['B7'] = [SNB7.mean(), SNB7.max(), SNB7.min(), SNB7.std()]
    S2_SN['B8'] = [SNB8.mean(), SNB8.max(), SNB8.min(), SNB8.std()]
    S2_SN['B11'] = [SNB11.mean(), SNB11.max(), SNB11.min(), SNB11.std()]
    S2_SN['B12'] = [SNB12.mean(), SNB12.max(), SNB12.min(), SNB12.std()]

    S2_WAT = pd.DataFrame(columns=['B2','B3','B4','B5','B6','B7','B8','B11','B12'], index=['mean', 'max', 'min', 'std'])
    S2_WAT['B2'] = [WATB2.mean(), WATB2.max(), WATB2.min(),WATB2.std()]
    S2_WAT['B3'] = [WATB3.mean(), WATB3.max(), WATB3.min(), WATB3.std()]
    S2_WAT['B4'] = [WATB4.mean(), WATB4.max(), WATB4.min(), WATB4.std()]
    S2_WAT['B5'] = [WATB5.mean(), WATB5.max(), WATB5.min(), WATB5.std()]
    S2_WAT['B6'] = [WATB6.mean(), WATB6.max(), WATB6.min(), WATB6.std()]
    S2_WAT['B7'] = [WATB7.mean(), WATB7.max(), WATB7.min(), WATB7.std()]
    S2_WAT['B8'] = [WATB8.mean(), WATB8.max(), WATB8.min(), WATB8.std()]
    S2_WAT['B11'] = [WATB11.mean(), WATB11.max(), WATB11.min(), WATB11.std()]
    S2_WAT['B12'] = [WATB12.mean(), WATB12.max(), WATB12.min(), WATB12.std()]

    S2_CC = pd.DataFrame(columns=['B2','B3','B4','B5','B6','B7','B8','B11','B12'], index=['mean', 'max', 'min', 'std'])
    S2_CC['B2'] = [CCB2.mean(), CCB2.max(), CCB2.min(),CCB2.std()]
    S2_CC['B3'] = [CCB3.mean(), CCB3.max(), CCB3.min(), CCB3.std()]
    S2_CC['B4'] = [CCB4.mean(), CCB4.max(), CCB4.min(), CCB4.std()]
    S2_CC['B5'] = [CCB5.mean(), CCB5.max(), CCB5.min(), CCB5.std()]
    S2_CC['B6'] = [CCB6.mean(), CCB6.max(), CCB6.min(), CCB6.std()]
    S2_CC['B7'] = [CCB7.mean(), CCB7.max(), CCB7.min(), CCB7.std()]
    S2_CC['B8'] = [CCB8.mean(), CCB8.max(), CCB8.min(), CCB8.std()]
    S2_CC['B11'] = [CCB11.mean(), CCB11.max(), CCB11.min(), CCB11.std()]
    S2_CC['B12'] = [CCB12.mean(), CCB12.max(), CCB12.min(), CCB12.std()]

    S2_CI = pd.DataFrame(columns=['B2','B3','B4','B5','B6','B7','B8','B11','B12'], index=['mean', 'max', 'min', 'std'])
    S2_CI['B2'] = [CIB2.mean(), CIB2.max(), CIB2.min(), CIB2.std()]
    S2_CI['B3'] = [CIB3.mean(), CIB3.max(), CIB3.min(), CIB3.std()]
    S2_CI['B4'] = [CIB4.mean(), CIB4.max(), CIB4.min(), CIB4.std()]
    S2_CI['B5'] = [CIB5.mean(), CIB5.max(), CIB5.min(), CIB5.std()]
    S2_CI['B6'] = [CIB6.mean(), CIB6.max(), CIB6.min(), CIB6.std()]
    S2_CI['B7'] = [CIB7.mean(), CIB7.max(), CIB7.min(), CIB7.std()]
    S2_CI['B8'] = [CIB8.mean(), CIB8.max(), CIB8.min(), CIB8.std()]
    S2_CI['B11'] = [CIB11.mean(), CIB11.max(), CIB11.min(), CIB11.std()]
    S2_CI['B12'] = [CIB12.mean(), CIB12.max(), CIB12.min(), CIB12.std()]

    S2_LA = pd.DataFrame(columns=['B2','B3','B4','B5','B6','B7','B8','B11','B12'], index=['mean', 'max', 'min', 'std'])
    S2_LA['B2'] = [LAB2.mean(), LAB2.max(), LAB2.min(), LAB2.std()]
    S2_LA['B3'] = [LAB3.mean(), LAB3.max(), LAB3.min(), LAB3.std()]
    S2_LA['B4'] = [LAB4.mean(), LAB4.max(), LAB4.min(), LAB4.std()]
    S2_LA['B5'] = [LAB5.mean(), LAB5.max(), LAB5.min(), LAB5.std()]
    S2_LA['B6'] = [LAB6.mean(), LAB6.max(), LAB6.min(), LAB6.std()]
    S2_LA['B7'] = [LAB7.mean(), LAB7.max(), LAB7.min(), LAB7.std()]
    S2_LA['B8'] = [LAB8.mean(), LAB8.max(), LAB8.min(), LAB8.std()]
    S2_LA['B11'] = [LAB11.mean(), LAB11.max(), LAB11.min(), LAB11.std()]
    S2_LA['B12'] = [LAB12.mean(), LAB12.max(), LAB12.min(), LAB12.std()]

    S2_HA = pd.DataFrame(columns=['B2','B3','B4','B5','B6','B7','B8','B11','B12'], index=['mean', 'max', 'min', 'std'])
    S2_HA['B2'] = [HAB2.mean(), HAB2.max(), HAB2.min(), HAB2.std()]
    S2_HA['B3'] = [HAB3.mean(), HAB3.max(), HAB3.min(), HAB3.std()]
    S2_HA['B4'] = [HAB4.mean(), HAB4.max(), HAB4.min(), HAB4.std()]
    S2_HA['B5'] = [HAB5.mean(), HAB5.max(), HAB5.min(), HAB5.std()]
    S2_HA['B6'] = [HAB6.mean(), HAB6.max(), HAB6.min(), HAB6.std()]
    S2_HA['B7'] = [HAB7.mean(), HAB7.max(), HAB7.min(), HAB7.std()]
    S2_HA['B8'] = [HAB8.mean(), HAB8.max(), HAB8.min(), HAB8.std()]
    S2_HA['B11'] = [HAB11.mean(), HAB11.max(), HAB11.min(), HAB11.std()]
    S2_HA['B12'] = [HAB12.mean(), HAB12.max(), HAB12.min(), HAB12.std()]

    # flush disk
    daB2 = None
    daB3 = None
    daB4 = None
    daB5 = None
    daB6 = None
    daB7 = None
    daB8 = None
    daB11 = None
    daB12 = None


    return S2_HA, S2_LA, S2_CI, S2_CC, S2_SN, S2_WAT


def ASD():

    hcrf_master = pd.read_csv('/home/joe/Code/IceSurfClassifiers/Training_Data/HCRF_master_machine_snicar.csv')
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

    ASD_HA =pd.DataFrame(columns = ['B02','B03','B04','B05','B06','B07','B08','B11','B12'],index = ['mean','max','min','std'] )
    ASD_HAtemp = X[X['label']=='HA']
    ASD_HAtemp.columns=['B02','B03','B04','B05','B06','B07','B08','B11','B12','label']
    ASD_HA.loc['mean',:]=ASD_HAtemp.mean()
    ASD_HA.loc['max',:]=ASD_HAtemp.max()
    ASD_HA.loc['min',:] =ASD_HAtemp.min()
    ASD_HA.loc['std',:]=ASD_HAtemp.std()

    ASD_LA =pd.DataFrame(columns = ['B02','B03','B04','B05','B06','B07','B08','B11','B12'],index = ['mean','max','min','std'] )
    ASD_LAtemp = X[X['label']=='LA']
    ASD_LAtemp.columns=['B02','B03','B04','B05','B06','B07','B08','B11','B12','label']
    ASD_LA.loc['mean',:]=ASD_LAtemp.mean()
    ASD_LA.loc['max',:]=ASD_LAtemp.max()
    ASD_LA.loc['min',:] =ASD_LAtemp.min()
    ASD_LA.loc['std',:]=ASD_LAtemp.std()

    ASD_CI =pd.DataFrame(columns = ['B02','B03','B04','B05','B06','B07','B08','B11','B12'],index = ['mean','max','min','std'] )
    ASD_CItemp = X[X['label']=='CI']
    ASD_CItemp.columns=['B02','B03','B04','B05','B06','B07','B08','B11','B12','label']
    ASD_CI.loc['mean',:]=ASD_CItemp.mean()
    ASD_CI.loc['max',:]=ASD_CItemp.max()
    ASD_CI.loc['min',:] =ASD_CItemp.min()
    ASD_CI.loc['std',:]=ASD_CItemp.std()

    ASD_CC =pd.DataFrame(columns = ['B02','B03','B04','B05','B06','B07','B08','B11','B12'],index = ['mean','max','min','std'] )
    ASD_CCtemp = X[X['label']=='CC']
    ASD_CCtemp.columns=['B02','B03','B04','B05','B06','B07','B08','B11','B12','label']
    ASD_CC.loc['mean',:]=ASD_CCtemp.mean()
    ASD_CC.loc['max',:]=ASD_CCtemp.max()
    ASD_CC.loc['min',:] =ASD_CCtemp.min()
    ASD_CC.loc['std',:]=ASD_CCtemp.std()

    ASD_WAT =pd.DataFrame(columns = ['B02','B03','B04','B05','B06','B07','B08','B11','B12'],index = ['mean','max','min','std'] )
    ASD_WATtemp = X[X['label']=='WAT']
    ASD_WATtemp.columns=['B02','B03','B04','B05','B06','B07','B08','B11','B12','label']
    ASD_WAT.loc['mean',:]=ASD_WATtemp.mean()
    ASD_WAT.loc['max',:]=ASD_WATtemp.max()
    ASD_WAT.loc['min',:] =ASD_WATtemp.min()
    ASD_WAT.loc['std',:]=ASD_WATtemp.std()

    ASD_SN =pd.DataFrame(columns = ['B02','B03','B04','B05','B06','B07','B08','B11','B12'],index = ['mean','max','min','std'] )
    ASD_SNtemp = X[X['label']=='SN']
    ASD_SNtemp.columns=['B02','B03','B04','B05','B06','B07','B08','B11','B12','label']
    ASD_SN.loc['mean',:]=ASD_SNtemp.mean()
    ASD_SN.loc['max',:]=ASD_SNtemp.max()
    ASD_SN.loc['min',:] =ASD_SNtemp.min()
    ASD_SN.loc['std',:]=ASD_SNtemp.std()


### Repeat process for UAV wavelengths, create new dataframe to store 5 band reflectance values ###
    ###########################################################################################

    X_UAV = pd.DataFrame()

    X_UAV['R475'] = np.array(HA_hcrf.iloc[125])
    X_UAV['R560'] = np.array(HA_hcrf.iloc[210])
    X_UAV['R668'] = np.array(HA_hcrf.iloc[318])
    X_UAV['R717'] = np.array(HA_hcrf.iloc[367])
    X_UAV['R840'] = np.array(HA_hcrf.iloc[490])

    X_UAV['label'] = 'HA'

    Y_UAV = pd.DataFrame()
    Y_UAV['R475'] = np.array(LA_hcrf.iloc[125])
    Y_UAV['R560'] = np.array(LA_hcrf.iloc[210])
    Y_UAV['R668'] = np.array(LA_hcrf.iloc[318])
    Y_UAV['R717'] = np.array(LA_hcrf.iloc[367])
    Y_UAV['R840'] = np.array(LA_hcrf.iloc[490])

    Y_UAV['label'] = 'LA'

    Z_UAV = pd.DataFrame()

    Z_UAV['R475'] = np.array(CI_hcrf.iloc[125])
    Z_UAV['R560'] = np.array(CI_hcrf.iloc[210])
    Z_UAV['R668'] = np.array(CI_hcrf.iloc[318])
    Z_UAV['R717'] = np.array(CI_hcrf.iloc[367])
    Z_UAV['R840'] = np.array(CI_hcrf.iloc[490])

    Z_UAV['label'] = 'CI'

    P_UAV = pd.DataFrame()

    P_UAV['R475'] = np.array(CC_hcrf.iloc[125])
    P_UAV['R560'] = np.array(CC_hcrf.iloc[210])
    P_UAV['R668'] = np.array(CC_hcrf.iloc[318])
    P_UAV['R717'] = np.array(CC_hcrf.iloc[367])
    P_UAV['R840'] = np.array(CC_hcrf.iloc[490])

    P_UAV['label'] = 'CC'

    Q_UAV = pd.DataFrame()
    Q_UAV['R475'] = np.array(WAT_hcrf.iloc[125])
    Q_UAV['R560'] = np.array(WAT_hcrf.iloc[210])
    Q_UAV['R668'] = np.array(WAT_hcrf.iloc[318])
    Q_UAV['R717'] = np.array(WAT_hcrf.iloc[367])
    Q_UAV['R840'] = np.array(WAT_hcrf.iloc[490])

    Q_UAV['label'] = 'WAT'

    R_UAV = pd.DataFrame()
    R_UAV['R475'] = np.array(SN_hcrf.iloc[125])
    R_UAV['R560'] = np.array(SN_hcrf.iloc[210])
    R_UAV['R668'] = np.array(SN_hcrf.iloc[318])
    R_UAV['R717'] = np.array(SN_hcrf.iloc[367])
    R_UAV['R840'] = np.array(SN_hcrf.iloc[490])

    R_UAV['label'] = 'SN'

    Zero = pd.DataFrame()
    Zero['R475'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R560'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R668'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R717'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Zero['R840'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    Zero['label'] = 'UNKNOWN'

    # Join dataframes into one continuous DF
    X_UAV = X_UAV.append(Y_UAV, ignore_index=True)
    X_UAV = X_UAV.append(Z_UAV, ignore_index=True)
    X_UAV = X_UAV.append(P_UAV, ignore_index=True)
    X_UAV = X_UAV.append(Q_UAV, ignore_index=True)
    X_UAV = X_UAV.append(R_UAV, ignore_index=True)
    X_UAV = X_UAV.append(Zero, ignore_index=True)


    ASD_HA_UAV =pd.DataFrame(columns = ['B1','B2','B3','B4','B5'],index = ['mean','max','min','std'] )
    ASD_HAtemp_UAV = X_UAV[X_UAV['label']=='HA']
    ASD_HAtemp_UAV.columns=['B1','B2','B3','B4','B5','label']
    ASD_HA_UAV.loc['mean',:]=ASD_HAtemp_UAV.mean()
    ASD_HA_UAV.loc['max',:]=ASD_HAtemp_UAV.max()
    ASD_HA_UAV.loc['min',:] =ASD_HAtemp_UAV.min()
    ASD_HA_UAV.loc['std',:]=ASD_HAtemp_UAV.std()

    ASD_LA_UAV =pd.DataFrame(columns = ['B1','B2','B3','B4','B5'],index = ['mean','max','min','std'] )
    ASD_LAtemp_UAV = X_UAV[X_UAV['label']=='LA']
    ASD_LAtemp_UAV.columns=['B1','B2','B3','B4','B5','label']
    ASD_LA_UAV.loc['mean',:]=ASD_LAtemp_UAV.mean()
    ASD_LA_UAV.loc['max',:]=ASD_LAtemp_UAV.max()
    ASD_LA_UAV.loc['min',:] =ASD_LAtemp_UAV.min()
    ASD_LA_UAV.loc['std',:]=ASD_LAtemp_UAV.std()

    ASD_CI_UAV =pd.DataFrame(columns = ['B1','B2','B3','B4','B5'],index = ['mean','max','min','std'] )
    ASD_CItemp_UAV = X_UAV[X_UAV['label']=='CI']
    ASD_CItemp_UAV.columns=['B1','B2','B3','B4','B5','label']
    ASD_CI_UAV.loc['mean',:]=ASD_CItemp_UAV.mean()
    ASD_CI_UAV.loc['max',:]=ASD_CItemp_UAV.max()
    ASD_CI_UAV.loc['min',:] =ASD_CItemp_UAV.min()
    ASD_CI_UAV.loc['std',:]=ASD_CItemp_UAV.std()

    ASD_CC_UAV =pd.DataFrame(columns = ['B1','B2','B3','B4','B5'],index = ['mean','max','min','std'] )
    ASD_CCtemp_UAV = X_UAV[X_UAV['label']=='CC']
    ASD_CCtemp_UAV.columns=['B1','B2','B3','B4','B5','label']
    ASD_CC_UAV.loc['mean',:]=ASD_CCtemp_UAV.mean()
    ASD_CC_UAV.loc['max',:]=ASD_CCtemp_UAV.max()
    ASD_CC_UAV.loc['min',:] =ASD_CCtemp_UAV.min()
    ASD_CC_UAV.loc['std',:]=ASD_CCtemp_UAV.std()

    ASD_WAT_UAV =pd.DataFrame(columns = ['B1','B2','B3','B4','B5'],index = ['mean','max','min','std'] )
    ASD_WATtemp_UAV = X_UAV[X_UAV['label']=='WAT']
    ASD_WATtemp_UAV.columns=['B1','B2','B3','B4','B5','label']
    ASD_WAT_UAV.loc['mean',:]=ASD_WATtemp_UAV.mean()
    ASD_WAT_UAV.loc['max',:]=ASD_WATtemp_UAV.max()
    ASD_WAT_UAV.loc['min',:] =ASD_WATtemp_UAV.min()
    ASD_WAT_UAV.loc['std',:]=ASD_WATtemp_UAV.std()

    ASD_SN_UAV =pd.DataFrame(columns = ['B1','B2','B3','B4','B5'],index = ['mean','max','min','std'] )
    ASD_SNtemp_UAV = X_UAV[X_UAV['label']=='SN']
    ASD_SNtemp_UAV.columns=['B1','B2','B3','B4','B5','label']
    ASD_SN_UAV.loc['mean',:]=ASD_SNtemp_UAV.mean()
    ASD_SN_UAV.loc['max',:]=ASD_SNtemp_UAV.max()
    ASD_SN_UAV.loc['min',:] =ASD_SNtemp_UAV.min()
    ASD_SN_UAV.loc['std',:]=ASD_SNtemp_UAV.std()


    return X, ASD_HA, ASD_CC, ASD_CI, ASD_LA, ASD_SN, ASD_WAT, ASD_HA_UAV, ASD_CC_UAV, ASD_CI_UAV, ASD_LA_UAV, ASD_SN_UAV, ASD_WAT_UAV

UAV_CI, UAV_CC, UAV_HA, UAV_LA, UAV_SN, UAV_WAT = UAV()
S2_HA, S2_LA, S2_CI, S2_CC, S2_SN = S2()
X, ASD_HA, ASD_CC, ASD_CI, ASD_LA, ASD_SN, ASD_WAT, ASD_HA_UAV, ASD_CC_UAV, ASD_CI_UAV, ASD_LA_UAV, ASD_SN_UAV, ASD_WAT_UAV = ASD()

plt.figure(),plt.grid(None)
x = np.arange(0, 9, 1)
plt.scatter(x, S2_HA.loc['mean', 'B2':'B12'].values, marker='o', s=50, color = 'g', label='S2 HA')
plt.scatter(x, S2_LA.loc['mean', 'B2':'B12'].values, marker='o', s=50, color = 'r', label='S2 LA')
plt.scatter(x, S2_CI.loc['mean', 'B2':'B12'].values, marker='o', s=50, color = 'b', label='S2 CI')
plt.scatter(x, S2_CC.loc['mean', 'B2':'B12'].values, marker='o', s=50, color = 'k', label='S2 CC')
plt.scatter(x, ASD_HA.loc['mean', 'B02':'B12'].values, marker='s', s=50, color = 'g', label='ASD HA')
plt.scatter(x, ASD_LA.loc['mean', 'B02':'B12'].values, marker='s', s=50, color = 'r', label='ASD LA')
plt.scatter(x, ASD_CI.loc['mean', 'B02':'B12'].values, marker='s', s=50, color = 'b', label='ASD CI')
plt.scatter(x, ASD_CC.loc['mean', 'B02':'B12'].values, marker='s', s=50, color = 'k', label='ASD CC')
xx = np.arange(0, 5, 1)
plt.scatter(xx, UAV_HA.loc['mean', 'B1':'B5'].values, marker='x', s=50, color = 'g', label='UAV HA')
plt.scatter(xx, UAV_LA.loc['mean', 'B1':'B5'].values, marker='x', s=50, color = 'r', label='UAV LA')
plt.scatter(xx, UAV_CI.loc['mean', 'B1':'B5'].values, marker='x', s=50, color = 'b', label='UAV CI')
plt.scatter(xx, UAV_CC.loc['mean', 'B1':'B5'].values, marker='x', s=50, color = 'k', label='UAV CC')
# plt.errorbar(x,UAV_WAT.loc['mean','B1':'B5'].values, yerr=UAV_WAT.loc['std','B1':'B5'], marker='o', label='WAT'),\
# plt.errorbar(x,UAV_SN.loc['mean','B1':'B5'].values, yerr=UAV_SN.loc['std','B1':'B5'], marker='o', label='SN'),
plt.legend(loc='best'),plt.ylim(0,1),plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9],labels=['B1','B2','B3','B4','B5','B6','B7','B8','B9'])
plt.xlabel('BAND ID'),plt.ylabel('Reflectance'),plt.grid(None),plt.ylim(0,1),plt.xlim(-1,9)
plt.show()



plt.figure(),plt.grid(None)
x = np.arange(0, 9, 1)
plt.errorbar(x, S2_HA.loc['mean', 'B2':'B12'].values, marker='o', yerr=S2_HA.loc['std','B2':'B12'], color = 'g', label='S2 HA')
plt.errorbar(x, S2_LA.loc['mean', 'B2':'B12'].values, marker='o', yerr = S2_LA.loc['std', 'B2':'B12'], color = 'r', label='S2 LA')
plt.errorbar(x, S2_CI.loc['mean', 'B2':'B12'].values, marker='o', yerr = S2_CI.loc['std', 'B2':'B12'],  color = 'b', label='S2 CI')
plt.errorbar(x, S2_CC.loc['mean', 'B2':'B12'].values, marker='o', yerr = S2_CC.loc['std', 'B2':'B12'], color = 'k', label='S2 CC')
#plt.errorbar(x, S2_WAT.loc['mean', 'B2':'B12'].values, marker='o', yerr = S2_WAT.loc['std', 'B2':'B12'], color = 'y', label='S2 WAT')

plt.errorbar(x, ASD_HA.loc['mean', 'B02':'B12'].values, marker='s', ls = '--', yerr=ASD_HA.loc['std','B02':'B12'], color = 'g', label='ASD HA')
plt.errorbar(x, ASD_LA.loc['mean', 'B02':'B12'].values, marker='s', ls = '--', yerr=ASD_LA.loc['std','B02':'B12'], color = 'r', label='ASD LA')
plt.errorbar(x, ASD_CI.loc['mean', 'B02':'B12'].values, marker='s', ls = '--', yerr=ASD_CI.loc['std','B02':'B12'], color = 'b', label='ASD CI')
plt.errorbar(x, ASD_CC.loc['mean', 'B02':'B12'].values, marker='s', ls = '--', yerr=ASD_CC.loc['std','B02':'B12'], color = 'k', label='ASD CC')
#plt.errorbar(x, ASD_WAT.loc['mean', 'B02':'B12'].values, marker='o', yerr = ASD_WAT.loc['std', 'B02':'B12'], color = 'y', label='S2 WAT')
plt.xlabel('BAND ID'),plt.ylabel('Reflectance'), plt.grid(None), plt.legend(), plt.title('ASD vs Sentinel 2')


plt.figure()
xx = np.arange(0, 5, 1)
plt.errorbar(xx, ASD_HA_UAV.loc['mean', 'B1':'B5'].values, marker='s', ls = '--', yerr=ASD_HA_UAV.loc['std','B1':'B5'], color = 'g', label='ASD HA')
plt.errorbar(xx, ASD_LA_UAV.loc['mean', 'B1':'B5'].values, marker='s', ls = '--', yerr=ASD_LA_UAV.loc['std','B1':'B5'], color = 'r', label='ASD LA')
plt.errorbar(xx, ASD_CI_UAV.loc['mean', 'B1':'B5'].values, marker='s', ls = '--', yerr=ASD_CI_UAV.loc['std','B1':'B5'], color = 'b', label='ASD CI')
plt.errorbar(xx, ASD_CC_UAV.loc['mean', 'B1':'B5'].values, marker='s', ls = '--', yerr=ASD_CC_UAV.loc['std','B1':'B5'], color = 'k', label='ASD CC')
#plt.errorbar(xx, ASD_WAT_UAV.loc['mean', 'B1':'B5'].values, marker='s', ls = '--', yerr=ASD_WAT_UAV.loc['std','B1':'B5'], color = 'y', label='ASD WAT')
plt.errorbar(xx, ASD_SN_UAV.loc['mean', 'B1':'B5'].values, marker='s', ls = '--', yerr=ASD_SN_UAV.loc['std','B1':'B5'], color = 'y', label='ASD SN')

plt.errorbar(xx, UAV_HA.loc['mean', 'B1':'B5'].values, marker='x', yerr=UAV_HA.loc['std','B1':'B5'], color = 'g', label='UAV HA')
plt.errorbar(xx, UAV_LA.loc['mean', 'B1':'B5'].values, marker='x', yerr=UAV_LA.loc['std','B1':'B5'], color = 'r', label='UAV LA')
plt.errorbar(xx, UAV_CI.loc['mean', 'B1':'B5'].values, marker='x', yerr=UAV_CI.loc['std','B1':'B5'], color = 'b', label='UAV CI')
plt.errorbar(xx, UAV_CC.loc['mean', 'B1':'B5'].values, marker='x', yerr=UAV_CC.loc['std','B1':'B5'], color = 'k', label='UAV CC')
#plt.errorbar(xx, UAV_WAT.loc['mean', 'B1':'B5'].values, marker='x', yerr=UAV_WAT.loc['std','B1':'B5'], color = 'y', label='UAV WAT')
plt.errorbar(xx,UAV_SN.loc['mean','B1':'B5'].values, yerr=UAV_SN.loc['std','B1':'B5'], marker='o', color = 'y', label='UAV SN'),
plt.legend(loc='best'),plt.xticks(ticks=[0,1,2,3,4,5],labels=['B1','B2','B3','B4','B5'])
plt.xlabel('BAND ID'),plt.ylabel('Reflectance'),plt.grid(None),plt.ylim(0,1),plt.xlim(0,4)
plt.title('ASD vs UAV')
plt.show()