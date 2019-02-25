"""
*** INFO ***

code written by Joseph Cook (University of Sheffield), 2018. Correspondence to joe.cook@sheffield.ac.uk

*** OVERVIEW ***

This code loads a trained classifier from a .pkl file, then deploys it to classify Sentinel 2
imagery into various surface categories:

Water, Snow, Clean Ice, Cryoconite, Light Algal Bloom, Heavy Algal Bloom

The result is a classified ice surface map. The coverage statistics are calculated and reported.
The albedo of each pixel is calculated from the multispectral reflectance using Liang et al's (2002) narrowband to
broadband conversion formula. The result is an albedo map of the ice surface.

Both the classified map and albedo map are trimmed to remove non-ice areas using the Greenland Ice Mapping Project mask
before spatial stats are calculated.

This script repeats this process for numerous images from the W coast of Greenland. The tiles used are from the Western
margin of the ice sheet on 21st July 2016 and **only tiles without significant cloud cover** are used.


*** PREREQUISITES ***

1) A trained model pickled and saved to the working directory (the individual image processing scripts can be used to
train, pickle and save classifiers)

2) Sentinel-2 band images. Folders containing Level 1C products can be downloaded from Earthexplorer.usgs.gov
These must be converted from level 1C to Level 2A (i.e. corrected for atmospheric effects and reprojected to a
consistent 20m ground resolution) using the ESA command line tool Sen2Cor.

This requires downloading the Sen2Cor software and running from the command line. Instructions are available here:
https://forum.step.esa.int/t/sen2cor-2-4-0-stand-alone-installers-how-to-install/6908

Sen2Cor details:

L2A processor path =  '/home/joe/Sen2Cor/Sen2Cor-02.05.05-Linux64/bin/L2A_Process'
Default configuration file = '/home/joe/sen2cor/2.5/cfg/L2A_GIPP.xml'

With file downloaded from EarthExplorer on desktop, L1C to L2A processing achieved using optional function
process_L1C_to_L2A(). This iterates through the files named in L1Cfiles and saves processed files to the working directory.
These files were downloaded from SentinelHub - there were processing problems with the same files downloaded from
earthexplorer.usgs.gov.

The processed jp2s are then used as input data in this script.

3) The GIMP mask downloaded from https://nsidc.org/data/nsidc-0714/versions/1 must be saved to the working directory.
Ensure the downloaded tile is the correct one for the section of ice sheet being examined. In this code the sections
of the GIMP mask (tiles 1_0 1_1, 1_2 1_3 and 1_4) covering the western coast have beene merged into one single mosaic.
This was achieved using the following gdal commands.

source activate IceSurfClassifiers
gdal_merge.py GimpIceMask_15m_tile1_1_v1_1.tif GimpIceMask_15m_tile1_2_v1_1.tif GimpIceMask_15m_tile1_3_v1_1.tif -o
merged_mask.tif

The merged mask is then reprojected to match the sentinel 2 tile and cropped to the relevant area in the function
format_mask().


*** FUNCTIONS***

This script is divided into several functions. The first function (set_paths) organises the paths to each of the load
and save locations. The area labels are also set so that the script can keep track of which figure/savefile belongs
with each tile.

The second function (format_mask) reprojects the GIMP mask to an identical coordinate system, pixel size and spatial
extent to the Sentinel 2 images and returns a Boolean numpy array that will later be used to mask out non-ice areas
of the classificed map and albedo map

The third function (load_model_and_images) simply loads in the classifier as clf and processes the L2A images into
numpy arrays

The fourth function (classify_images) applies the trained classifier to the sentinel 2 images and masks out non-ice areas, then applies
Liang et al(2002) narrowband to broadband conversion formula, producing a NetCDF file containing all the data arrays and
metadata along with a plot of each map.

The fifth function (albedo_report) calculates spatial statistics for the classified surface and albedo maps. The final
function (merged_albedo_report) appends the albedo data from all the sites into one dataset and provides coverage and albedo statistics for the
total area imaged.

"""


###########################################################################################
############################# IMPORT MODULES #########################################

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import xarray as xr
from osgeo import gdal, osr
import georaster
import os

# matplotlib settings: use ggplot style and turn interactive mode off
mpl.style.use('ggplot')
plt.ioff()

# DEFINE FUNCTIONS

def process_L1C_to_L2A(L1C_path, L1Cfiles):
    """
    This function is takes the downloaded L1C products from SentinelHub and converts them to L2A product using Sen2Cor.
    This is achieved as a batch job by iterating through the file names (L1Cfiles) stored in L1Cpath
    Running this script will save the L2A products to the working directory

    :return: None

    """
    for L1C in L1Cfiles:
        cmd = str(
            '/home/joe/Sen2Cor/Sen2Cor-02.05.05-Linux64/bin/L2A_Process ' + L1C_path + '{} --resolution=20'.format(
                L1C))
        os.system(cmd)

    return

def set_paths(year = 2017):

    """
    function sets load and save paths
    paths are different for local or virtual machine

    :param virtual_machine: Boolean to control paths depending whether script is run locally or on VM
    :return: paths and some key variables



    """
    if year==2016:
        img_paths = ['/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2016/S2A_L2A_ILL2/GRANULE/S2A_USER_MSI_L2A_TL_MTI__20160721T202530_A005642_T22WEB_N02_04/IMG_DATA/R20m/',
                     '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2016/S2_L2A_ILL1/GRANULE/S2A_USER_MSI_L2A_TL_MTI__20160721T202530_A005642_T22WEA_N02_04/IMG_DATA/R20m/',
                     '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2016/S2_L2A_ILL3/GRANULE/S2A_USER_MSI_L2A_TL_MTI__20160721T202530_A005642_T22WEC_N02_04/IMG_DATA/R20m/',
                     '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2016/S2_L2A_KGR/GRANULE/L2A_T22WEV_A005642_20160721T151913/IMG_DATA/R20m/',
                     '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2016/T21XWB/GRANULE/L2A_T21XWB_A005714_20160726T160905/IMG_DATA/R20m/',
                     '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2016/T22_WES/GRANULE/L2A_T22WES_A005699_20160725T145918/IMG_DATA/R20m']



        img_stubs = ['S2A_USER_MSI_L2A_TL_MTI__20160721T202530_A005642_T22WEB_', 'S2A_USER_MSI_L2A_TL_MTI__20160721T202530_A005642_T22WEA_', 'S2A_USER_MSI_L2A_TL_MTI__20160721T202530_A005642_T22WEC_','L2A_T22WEV_20160721T151912_', 'L2A_T21XWB_20160726T160902_','L2A_T22WES_20160725T145922_']
        cloudmaskpaths =['/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2016/S2A_L2A_ILL2/GRANULE/S2A_USER_MSI_L2A_TL_MTI__20160721T202530_A005642_T22WEB_N02_04/QI_DATA/S2A_USER_CLD_L2A_TL_MTI__20160721T202530_A005642_T22WEB_20m.jp2',
        '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2016/S2_L2A_ILL1/GRANULE/S2A_USER_MSI_L2A_TL_MTI__20160721T202530_A005642_T22WEA_N02_04/QI_DATA/S2A_USER_CLD_L2A_TL_MTI__20160721T202530_A005642_T22WEA_20m.jp2',
        '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2016/S2_L2A_ILL3/GRANULE/S2A_USER_MSI_L2A_TL_MTI__20160721T202530_A005642_T22WEC_N02_04/QI_DATA/S2A_USER_CLD_L2A_TL_MTI__20160721T202530_A005642_T22WEC_20m.jp2',
        '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2016/S2_L2A_KGR/GRANULE/L2A_T22WEV_A005642_20160721T151913/QI_DATA/L2A_T22WEV_20160721T151912_CLD_20m.jp2',
        '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2016/T21XWB/GRANULE/L2A_T21XWB_A005714_20160726T160905/QI_DATA/L2A_T21XWB_20160726T160902_CLD_20m.jp2',
        '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2016/T22_WES/GRANULE/L2A_T22WES_A005699_20160725T145918/QI_DATA/L2A_T22WES_20160725T145922_CLD_20m.jp2']





    elif year == 2017:
        img_paths = [
            '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2017/S2A_MSIL2A_20170726T151911_N0205_R068_T22WEA_20170726T151917.SAFE/GRANULE/L2A_T22WEA_A010933_20170726T151917/IMG_DATA/R20m/',
            '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2017/S2A_MSIL2A_20170726T151911_N0205_R068_T22WED_20170726T151917.SAFE/GRANULE/L2A_T22WED_A010933_20170726T151917/IMG_DATA/R20m/',
            '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2017/S2A_MSIL2A_20170724T161901_N0205_R040_T21XWB_20170724T162148.SAFE/GRANULE/L2A_T21XWB_A010905_20170724T162148/IMG_DATA/R20m/',
            '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2017/S2A_MSIL2A_20170726T151911_N0205_R068_T22WEC_20170726T151917.SAFE/GRANULE/L2A_T22WEC_A010933_20170726T151917/IMG_DATA/R20m/',
            '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2017/S2A_MSIL2A_20170726T151911_N0205_R068_T22WEV_20170726T151917.SAFE/GRANULE/L2A_T22WEV_A010933_20170726T151917/IMG_DATA/R20m/']
        img_stubs = ['L2A_T22WEA_20170726T151911_', 'L2A_T22WED_20170726T151911_', 'L2A_T21XWB_20170724T161901_', 'L2A_T22WEC_20170726T151911_', 'L2A_T22WEV_20170726T151911_']
        cloudmaskpaths =['/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/S2A_MSIL2A_20170726T151911_N0205_R068_T22WEA_20170726T151917.SAFE/GRANULE/L2A_T22WEA_A010933_20170726T151917/QI_DATA/L2A_T22WEA_20170726T151911_CLD_20m.jp2',
            '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2017/S2A_MSIL2A_20170726T151911_N0205_R068_T22WED_20170726T151917.SAFE/GRANULE/L2A_T22WED_A010933_20170726T151917/QI_DATA/L2A_T22WED_20170726T151911_CLD_20m.jp2',
            '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2017/S2A_MSIL2A_20170724T161901_N0205_R040_T21XWB_20170724T162148.SAFE/GRANULE/L2A_T21XWB_A010905_20170724T162148/QI_DATA/L2A_T21XWB_20170724T161901_CLD_20m.jp2',
            '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2017/S2A_MSIL2A_20170726T151911_N0205_R068_T22WEC_20170726T151917.SAFE/GRANULE/L2A_T22WEC_A010933_20170726T151917/QI_DATA/L2A_T22WEC_20170726T151911_CLD_20m.jp2',
            '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2_L2A/2017/S2A_MSIL2A_20170726T151911_N0205_R068_T22WEV_20170726T151917.SAFE/GRANULE/L2A_T22WEV_A010933_20170726T151917/QI_DATA/L2A_T22WEV_20170726T151911_CLD_20m.jp2']

    cloudProbThreshold = 50
    savefig_path = '/home/joe/Code/IceSurfClassifiers/Sentinel_Outputs/'

    # paths for format_mask()
    Icemask_in = '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/Mask/merged_mask.tif'
    Icemask_out = '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/Mask/GIMP_MASK.nc'

    pickle_path = '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/Sentinel2_classifier.pkl'
    area_labels = ['T22WEA','T22WED','T21XWB','T22WEC', 'T22WEV']
    masterDF = pd.DataFrame(columns=(['pred','albedo']))

    return savefig_path, img_paths, img_stubs, Icemask_in, Icemask_out, area_labels, masterDF, pickle_path, cloudmaskpaths, cloudProbThreshold


def format_mask(img_path, img_stub, Icemask_in, Icemask_out, cloudmaskpath, cloudProbThreshold):
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
    mask = gdal.Open(Icemask_in)

    mask_proj = mask.GetProjection()
    mask_geotrans = mask.GetGeoTransform()
    data_type = mask.GetRasterBand(1).DataType
    n_bands = mask.RasterCount

    Sentinel = gdal.Open(img_path+img_stub+'B02_20m.jp2')

    Sentinel_proj = Sentinel.GetProjection()
    Sentinel_geotrans = Sentinel.GetGeoTransform()
    w = Sentinel.RasterXSize
    h = Sentinel.RasterYSize

    mask_filename = Icemask_out
    new_mask = gdal.GetDriverByName('GTiff').Create(mask_filename,
                                                     w, h, n_bands, data_type)
    new_mask.SetGeoTransform(Sentinel_geotrans)
    new_mask.SetProjection(Sentinel_proj)

    gdal.ReprojectImage(mask, new_mask, mask_proj,
                        Sentinel_proj, gdal.GRA_NearestNeighbour)

    new_mask = None  # Flush disk

    maskxr = xr.open_rasterio(Icemask_out)
    mask_squeezed = xr.DataArray.squeeze(maskxr,'band')
    Icemask = xr.DataArray(mask_squeezed.values)

    # set up second mask for clouds
    Cloudmask = xr.open_rasterio(cloudmaskpath)
    Cloudmask = xr.DataArray.squeeze(Cloudmask,'band')
    # set pixels where probability of cloud < threshold to 0
    Cloudmask = Cloudmask.where(Cloudmask.values <= cloudProbThreshold, 0)

    return Icemask, Cloudmask


def load_model_and_images(Icemask, Cloudmask, img_path, img_stub, pickle_path):
    """
    function loads classifier from file and loads L2A image into numpy NDarray

    :param img_path: path to S2 L2A image saved as NetCDF
    :return: clf: classifier loaded in from .pkl file; S2vals: 9D numpy array containing pixel reflectance values

    """

    # Sentinel 2 dataset
    # create xarray dataset with all bands loaded from jp2s. Values are reflectance.

    daB2 = xr.open_rasterio(str(img_path + img_stub + 'B02_20m.jp2'), chunks={'x': 2000, 'y': 2000})
    daB3 = xr.open_rasterio(str(img_path + img_stub + 'B03_20m.jp2'), chunks={'x': 2000, 'y': 2000})
    daB4 = xr.open_rasterio(str(img_path + img_stub + 'B04_20m.jp2'), chunks={'x': 2000, 'y': 2000})
    daB5 = xr.open_rasterio(str(img_path + img_stub + 'B05_20m.jp2'), chunks={'x': 2000, 'y': 2000})
    daB6 = xr.open_rasterio(str(img_path + img_stub + 'B06_20m.jp2'), chunks={'x': 2000, 'y': 2000})
    daB7 = xr.open_rasterio(str(img_path + img_stub + 'B07_20m.jp2'), chunks={'x': 2000, 'y': 2000})
    daB8 = xr.open_rasterio(str(img_path + img_stub + 'B8A_20m.jp2'), chunks={'x': 2000, 'y': 2000})
    daB11 = xr.open_rasterio(str(img_path + img_stub + 'B11_20m.jp2'), chunks={'x': 2000, 'y': 2000})
    daB12 = xr.open_rasterio(str(img_path + img_stub + 'B12_20m.jp2'), chunks={'x': 2000, 'y': 2000})

    daB2 = xr.DataArray.squeeze(daB2, dim='band')
    daB3 = xr.DataArray.squeeze(daB3, dim='band')
    daB4 = xr.DataArray.squeeze(daB4, dim='band')
    daB5 = xr.DataArray.squeeze(daB5, dim='band')
    daB6 = xr.DataArray.squeeze(daB6, dim='band')
    daB7 = xr.DataArray.squeeze(daB7, dim='band')
    daB8 = xr.DataArray.squeeze(daB8, dim='band')
    daB11 = xr.DataArray.squeeze(daB11, dim='band')
    daB12 = xr.DataArray.squeeze(daB12, dim='band')

    S2vals = xr.Dataset({'B02': (('y', 'x'), daB2.values / 10000), 'B03': (('y', 'x'), daB3.values / 10000),
                         'B04': (('y', 'x'), daB4.values / 10000), 'B05': (('y', 'x'), daB5.values / 10000),
                         'B06': (('y', 'x'), daB6.values / 10000),
                         'B07': (('y', 'x'), daB7.values / 10000), 'B08': (('y', 'x'), daB8.values / 10000),
                         'B11': (('y', 'x'), daB11.values / 10000),
                         'B12': (('y', 'x'), daB12.values / 10000), 'Icemask': (('y', 'x'), Icemask),
                         'Cloudmask': (('y', 'x'), Cloudmask)})

    S2vals.to_netcdf(savefig_path + "S2vals.nc", mode='w')
    S2vals = None
    daB2 = None
    daB3 = None
    daB4 = None
    daB5 = None
    daB6 = None
    daB7 = None
    daB8 = None
    daB11 = None
    daB12 = None

    #load pickled model
    clf = joblib.load(pickle_path)

    return clf


def ClassifyImages(clf, img_path, img_stub, area_label, savefigs=False):

    with xr.open_dataset(savefig_path + "S2vals.nc",chunks={'x':2000,'y':2000}) as S2vals:
        # Set index for reducing data
        band_idx = pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9], name='bands')

        # concatenate the bands into a single dimension ('bands_idx') in the data array
        concat = xr.concat([S2vals.B02, S2vals.B03, S2vals.B04, S2vals.B05, S2vals.B06, S2vals.B07,
                            S2vals.B08, S2vals.B11, S2vals.B12], band_idx)

        # stack the values into a 1D array
        stacked = concat.stack(allpoints=['y', 'x'])

        # Transpose and rename so that DataArray has exactly the same layout/labels as the training DataArray.
        # mask out nan areas not masked out by GIMP
        stackedT = stacked.T
        stackedT = stackedT.rename({'allpoints': 'samples'})

        # apply classifier
        predicted = clf.predict(stackedT)

        # Unstack back to x,y grid
        predicted = predicted.unstack(dim='samples')

        #calculate albeod using Liang et al (2002) equation
        albedo = xr.DataArray(0.356 * (concat.values[1]) + 0.13 * (concat.values[3]) + 0.373 * \
                       (concat.values[6]) + 0.085 * (concat.values[7]) + 0.072 * (concat.values[8]) - 0.0018)

        #update mask so that both GIMP mask and areas not sampled by S2 but not masked by GIMP both = 0
        mask2 = (S2vals.Icemask.values ==1) & (concat.sum(dim='bands')>0) & (S2vals.Cloudmask.values == 0)

        # collate predicted map, albedo map and projection info into xarray dataset
        # 1) Retrieve projection info from S2 datafile and add to netcdf
        srs = osr.SpatialReference()
        srs.ImportFromProj4('+init=epsg:32622') # Get info for UTM zone 22N
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
        # see georaster docs at https:/media.readthedocs.org/pdf/georaster/latest/georaster.pdf
        S2 = georaster.SingleBandRaster(img_path + img_stub + 'B02_20m.jp2', load_data=False)
        lon, lat = S2.coordinates(latlon=True)
        S2 = None

        S2 = xr.open_rasterio(img_path + img_stub + 'B02_20m.jp2', chunks={'x': 2000, 'y': 2000})
        coords_geo = {'y': S2['y'], 'x': S2['x']}
        S2 = None

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
        predictedxr = xr.DataArray(predicted.values, coords=coords_geo, dims=['y', 'x'])
        predictedxr = predictedxr.fillna(0)
        predictedxr = predictedxr.where(mask2>0)
        predictedxr.encoding = {'dtype': 'int16', 'zlib': True, '_FillValue': -9999}
        predictedxr.name = 'Surface Class'
        predictedxr.attrs['long_name'] = 'Surface classified using Random Forest'
        predictedxr.attrs['units'] = 'None'
        predictedxr.attrs[
            'key'] = 'Snow:1; Water:2; Cryoconite:3; Clean Ice:4; Light Algae:5; Heavy Algae:6'
        predictedxr.attrs['grid_mapping'] = 'UTM'

        # add albedo map array and add metadata
        albedoxr = xr.DataArray(albedo.values, coords=coords_geo, dims=['y', 'x'])
        albedoxr = albedoxr.fillna(0)
        albedoxr = albedoxr.where(mask2 > 0)
        albedoxr.encoding = {'dtype': 'int16', 'scale_factor': 0, 'zlib': True, '_FillValue': -9999}
        albedoxr.name = 'Surface albedo computed after Knap et al. (1999) narrowband-to-broadband conversion'
        albedoxr.attrs['units'] = 'dimensionless'
        albedoxr.attrs['grid_mapping'] = 'UTM'

        # collate data arrays into a dataset
        dataset = xr.Dataset({

            'classified': (['x', 'y'], predictedxr),
            'albedo':(['x','y'],albedoxr),
            'mask': (['x', 'y'], S2vals.Icemask.values),
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

        dataset.to_netcdf(savefig_path + "{}_Classification_and_Albedo_Data.nc".format(area_label), mode='w')
        dataset=None

    if savefigs:

        cmap1 = mpl.colors.ListedColormap(
            ['purple', 'white', 'royalblue', 'black', 'lightskyblue', 'mediumseagreen', 'darkgreen'])
        cmap1.set_under(color='white')  # make sure background is white
        cmap2 = plt.get_cmap('Greys_r')  # reverse greyscale for albedo
        cmap2.set_under(color='white')  # make sure background is white

        fig, axes = plt.subplots(figsize=(10,8), ncols=1, nrows=2)
        predictedxr.plot(ax=axes[0], cmap=cmap1, vmin=0, vmax=6)
        plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
        plt.title('Greenland Ice Sheet from Sentinel 2 classified using Random Forest Classifier (top) and albedo (bottom)')
        axes[0].set_aspect('equal')

        albedoxr.plot(ax=axes[1], cmap=cmap2, vmin=0, vmax=1)
        plt.ylabel('Latitude (UTM Zone 22N)'), plt.xlabel('Longitude (UTM Zone 22N)')
        axes[1].set_aspect('equal')
        plt.grid(None)

        fig.tight_layout()
        plt.savefig(str(savefig_path + "{}_Sentinel_Classified_Albedo.png".format(area_label)), dpi=300)
        plt.close()

    return

def albedo_report(masterDF, area_label, save_albedo_data=False):
    # match albedo to predicted class using indexes

    with xr.open_dataset(savefig_path + "{}_Classification_and_Albedo_Data.nc".format(area_label), chunks={'x':2000,'y':2000}) as dataset:
        predicted = np.array(dataset.classified.values).ravel()
        albedo = np.array(dataset.albedo.values).ravel()

        albedoDF = pd.DataFrame()
        albedoDF['pred'] = predicted
        albedoDF['albedo'] = albedo
        albedoDF = albedoDF.dropna()

        # coverage statistics
        HApercent = (albedoDF['pred'][albedoDF['pred'] == 6].count()) / (albedoDF['pred'][albedoDF['pred'] != 1].count())
        LApercent = (albedoDF['pred'][albedoDF['pred'] == 5].count()) / (albedoDF['pred'][albedoDF['pred'] != 1].count())
        CIpercent = (albedoDF['pred'][albedoDF['pred'] == 4].count()) / (albedoDF['pred'][albedoDF['pred'] != 1].count())
        CCpercent = (albedoDF['pred'][albedoDF['pred'] == 3].count()) / (albedoDF['pred'][albedoDF['pred'] != 1].count())
        WATpercent = (albedoDF['pred'][albedoDF['pred'] == 2].count()) / (albedoDF['pred'][albedoDF['pred'] != 1].count())
        SNpercent = (albedoDF['pred'][albedoDF['pred'] == 1].count()) / (albedoDF['pred'][albedoDF['pred'] != 0].count())

        if save_albedo_data:
            albedoDF.to_csv(savefig_path + 'RawAlbedoData_{}.csv'.format(area_label))
            albedoDF.groupby(['pred']).count().to_csv(savefig_path + 'Surface_Type_Counts_{}.csv'.format(area_label))
            albedoDF.groupby(['pred']).describe()['albedo'].to_csv(savefig_path + 'Albedo_summary_stats_{}.csv'.format(area_label))

        # report summary stats
        print('\n Surface type counts: \n', albedoDF.groupby(['pred']).count())
        print('\n Summary Statistics for ALBEDO of each surface type: \n', albedoDF.groupby(['pred']).describe()['albedo'])

        print('\n "Percent coverage by surface type: \n')
        print(' HA coverage = ', np.round(HApercent, 2) * 100, '%\n', 'LA coverage = ', np.round(LApercent, 2) * 100, '%\n',
              'CI coverage = ',
              np.round(CIpercent, 2) * 100, '%\n', 'CC coverage = ', np.round(CCpercent, 2) * 100, '%\n', 'SN coverage = ',
              np.round(SNpercent, 2) * 100, '%\n', 'WAT coverage = ', np.round(WATpercent, 2) * 100, '%\n',
              'Total Algal Coverage = ',
              np.round(HApercent + LApercent, 2) * 100)

        masterDF = masterDF.append(albedoDF, ignore_index=True)

    return masterDF

def merged_albedo_report(masterDF, print_stats_to_console = True, save_stats = True, save_raw = True):

    """
    function calculates summary stats for classifications and albedo for all the tiles merged into a single dataset
    NB this function is quite slow (>1min) because querying the large masterDF file using groupby is fairly
    computationally expensive

    :param masterDF: pandas dataframe to contain concatenated albedo data from all sites
    :param print_stats_to_console: Boolean to control whether albedo stats are printed to console
    :param save_stats: Boolean to control whether albedo summary stats are saved to file
    :param save_raw: Boolean to control whether the entire masterDF dataset is saved to file (slow)
    :return: none

    """

    # analyses and saves albedo and coverage stats for all tiles merged into single dataset
    # coverage statistics
    HApercent = (masterDF['pred'][masterDF['pred']==6].count()) / (masterDF['pred'][masterDF['pred']!=1].count())
    LApercent = (masterDF['pred'][masterDF['pred']==5].count()) / (masterDF['pred'][masterDF['pred']!=1].count())
    CIpercent = (masterDF['pred'][masterDF['pred']==4].count()) / (masterDF['pred'][masterDF['pred']!=1].count())
    CCpercent = (masterDF['pred'][masterDF['pred']==3].count()) / (masterDF['pred'][masterDF['pred']!=1].count())
    WATpercent = (masterDF['pred'][masterDF['pred'] == 2].count()) / (masterDF['pred'][masterDF['pred'] != 1].count())
    SNpercent = (masterDF['pred'][masterDF['pred']==1].count()) / (masterDF['pred'][masterDF['pred']!=0].count())

    percent_cover = pd.DataFrame(columns=(['class','percent_cover']))
    percent_cover['class'] = ['HA','LA','CI','CC','WAT','SN']
    percent_cover['percent_cover'] = [HApercent,LApercent,CIpercent,CCpercent,WATpercent,SNpercent]
    counts = masterDF.groupby(['pred']).count()
    summary = masterDF.groupby(['pred']).describe()['albedo']

    if print_stats_to_console:
        # report to console
        print('\n MERGED ALBEDO STATS:')
        print('\n Surface type counts: \n', counts)
        print('\n Summary Statistics for ALBEDO of merged tiles: \n',summary)
        print('\n "Percent coverage by surface type: \n')
        print(' HA coverage = ',np.round(HApercent,2)*100,'%\n','LA coverage = ',np.round(LApercent,2)*100,'%\n','CI coverage = ',
              np.round(CIpercent,2)*100,'%\n', 'CC coverage = ',np.round(CCpercent,2)*100,'%\n', 'SN coverage = ',
              np.round(SNpercent,2)*100,'%\n', 'WAT coverage = ', np.round(WATpercent,2)*100,'%\n', 'Total Algal Coverage = ',
              np.round(HApercent+LApercent,2)*100)

    if save_stats == True:
        percent_cover.to_csv(savefig_path+'percent_coverage_MERGED.csv')
        counts.to_csv(savefig_path+'Surface_Type_Counts_MERGED.csv')
        summary.to_csv(savefig_path+'Albedo_summary_statsMERGED.csv')

    if save_raw == True:
        masterDF.to_csv(savefig_path + 'RawAlbedoData_MERGED.csv')

    return


"""
************************************************************************
********************* RUN FUNCTIONS IN LOOP ****************************
************************************************************************

Set up loop to iterate through tiles
The functions set_paths() and merged_albedo_report() are outside of the loop
other functions are called iteratively

"""
# RUN FUNCTIONS

## Uncomment block to batch process L1C files to L2A

# L1C_path = '/data/home/tothepoles/PycharmProjects/IceSurfClassifiers/'
# L1Cfiles = ['S2A_MSIL1C_20170724T161901_N0205_R040_T21XWB_20170724T162148.SAFE',
#             'S2A_MSIL1C_20170726T151911_N0205_R068_T22WEA_20170726T151917.SAFE',
#             'S2A_MSIL1C_20170726T151911_N0205_R068_T22WEC_20170726T151917.SAFE',
#             'S2A_MSIL1C_20170726T151911_N0205_R068_T22WED_20170726T151917.SAFE',
#             'S2A_MSIL1C_20170726T151911_N0205_R068_T22WEV_20170726T151917.SAFE']
#
# process_L1C_to_L2A(L1C_path, L1Cfiles)


# ITERATE THROUGH IMAGES
import datetime
savefig_path,img_paths, img_stubs, Icemask_in, Icemask_out, area_labels, masterDF, pickle_path, \
cloudmaskpaths, cloudProbThreshold = set_paths(year=2016)
StartTime = datetime.datetime.now() #start timer

for i in np.arange(0,len(area_labels),1):

    area_label = area_labels[i]
    img_path = img_paths[i]
    img_stub = img_stubs[i]
    cloudmaskpath = cloudmaskpaths[i]

    #format mask
    Icemask, Cloudmask = format_mask(img_path, img_stub, Icemask_in, Icemask_out, cloudmaskpath, cloudProbThreshold)

    #create dataset
    clf = load_model_and_images(Icemask, Cloudmask, img_path, img_stub, pickle_path)

    # apply model to Sentinel2 image
    ClassifyImages(clf, img_path, img_stub, area_label, savefigs=True)

    # calculate spatial stats
    masterDF = albedo_report(masterDF, area_label, save_albedo_data=False)

    print('\n FINISHED RUNNING AREA: ','*** ', area_label, ' ****')

merged_albedo_report(masterDF, print_stats_to_console = True, save_stats = True, save_raw = False)

runTime = datetime.datetime.now() - StartTime # stop timer

print('Total time taken to run script = ', runTime)