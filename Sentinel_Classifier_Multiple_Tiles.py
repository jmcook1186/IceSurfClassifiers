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
Ensure the downloaded tile is the correct one for the section of ice sheet being examined. In this code the sections
of the GIMP mask (tiles 1_0 1_1, 1_2 1_3 and 1_4) covering the western coast have beene merged into one single mosaic.
This was achieved using the following gdal commands.

source activate IceSurfClassifiers
gdal_merge.py GimpIceMask_15m_tile1_1_v1_1.tif GimpIceMask_15m_tile1_2_v1_1.tif GimpIceMask_15m_tile1_3_v1_1.tif -o
merged_mask.tif

The merged mask is then reprojected to match the sentinel 2 tile and cropped to the relevant area in the function
format_mask()

A template for trimming the mask is required - these are specific to each tile, so #templates = #tiles.
An unprocessed, L1C band image is ideal. In this version I have arbitrarily chosen the *B02.jp2 image from each tile and
renamed as MaskTemplate_SITE.jp2 for clarity.


*** FUNCTIONS***

This script is divided into several functions. The first function (set_paths) organises the paths to each of the load
and save locations. The area labels are also set so that the script can keep track of which figure/savefile belongs
with each tile.

The second function (load_model_and_images) simply loads in the classifier as clf and processes the L2A images into
numpy arrays

The third function (format_mask) reprojects the GIMP mask to an identical coordinate system, pixel size and spatial
extent to the Sentinel 2 images and returns a Boolean numpy array that will later be used to mask out non-ice areas
of the classificed map and albedo map

The fourth function applies the trained classifier to the sentinel 2 images and masks out non-ice areas, then applies
Liang et al(2002) narrowband to broadband conversion formula, producing a NetCDF file containing all the data arrays and
metadata along with a plot of each map.

The final function calculates spatial statistics for the classified surface and albedo maps.

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

# matplotlib settings: use ggplot style and turn interactive mode off
mpl.style.use('ggplot')
plt.ioff()

# DEFINE FUNCTIONS
def set_paths(virtual_machine = False):

    """
    function sets load and save paths
    paths are different for local or virtual machine

    """

    if not virtual_machine:
        mask_path = '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/Mask/'
        savefig_path = '/home/joe/Code/IceSurfClassifiers/Sentinel_Outputs/'
        img_paths = ['/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2A_NetCDFs/KGR/',
                   '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2A_NetCDFs/ILL1/',
                   '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2A_NetCDFs/ILL2/',
                    '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/S2A_NetCDFs/ILL3/']
        # paths for format_mask()
        Sentinel_templates = [str(mask_path+'MaskTemplate_KGR.jp2'),str(mask_path+'MaskTemplate_ILL1.jp2'),
                             str(mask_path+'MaskTemplate_ILL2.jp2'),str(mask_path+'MaskTemplate_ILL3.jp2')]
        mask_in = '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/Mask/merged_mask.tif'
        mask_out = '/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/Mask/GIMP_MASK.nc'
        area_labels = ['KGR','ILL1','ILL2','ILL3']
        masterDF = pd.DataFrame(columns=(['pred','albedo']))

    else:
        # Virtual Machine
        # paths for create_dataset()
        savefig_path = '/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_Resources/Sentinel_Outputs/'
        img_path = '/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_Resources/S2A_NetCDFs/'

        # paths for format_mask()
        Sentinel_template = '/data/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_Resources/'
        mask_in = '/data/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_Resources/GIMP_MASK.tif'
        mask_out = '/data/home/tothepoles/PycharmProjects/IceSurfClassifiers/Sentinel_Resources/GIMP_MASK.nc'
        area_labels = ['KGR', 'ILL1', 'ILL2', 'ILL3']

    return savefig_path, img_paths, Sentinel_templates, mask_in, mask_out, area_labels, masterDF


def load_model_and_images(img_path):

    """
    function loads classifier from file and loads L2A image into numpy NDarray

    """

    S2vals = np.zeros([9,5490,5490])
    bands = ['02','03','04','05','05','07','08','11','12']
    for i in np.arange(0,len(bands),1):
        S2BX = xr.open_dataset(str(img_path+'B'+bands[i]+'.nc'))
        S2BXarr = S2BX.to_array()
        S2BXvals = np.array(S2BXarr.variable.values[1])
        S2BXvals = S2BXvals.astype(float)
        S2vals[i,:,:] = S2BXvals

    S2vals = S2vals/10000 # correct unit from S2 L2A data to reflectance between 0-1

    #load pickled model
    clf = joblib.load('/home/joe/Code/IceSurfClassifiers/Sentinel_Resources/Sentinel2_classifier.pkl')

    return S2vals, clf


def format_mask (Sentinel_template, mask_in, mask_out):
    """
    Function reprojects GIMP mask to dimensions, resolution and spatial coords of the S2 images, enabling
    Boolean masking of land-ice area.

    INPUTS:
    Sentinel = Sentinel image to use as projection template
    mask_in = file path to mask file
    mask_out = file path to save reprojected mask

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

    gdal.ReprojectImage(mask, new_mask, mask_proj,
                         Sentinel_proj, gdal.GRA_NearestNeighbour)
    new_mask = None  # Flush disk

    # open netCDF mask and extract values to numpy array
    maskxr = xr.open_dataset(mask_out)
    mask_array = np.array(maskxr.Band1.values)
    #replace nans with 0 to create binary numerical array
    nans = np.isnan(mask_array)
    mask_array[nans]=0

    return mask_array


def ClassifyImages(S2vals, clf, mask_array, area_label, savefigs=False, save_netcdf = False):
    """
    function deploys classifier to Sentinel 2 image and masks out non-ice areas

    """

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
                   'imagery of the Greenland Ice Sheet: {}'.format(area_label)

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
        dataset.to_netcdf(savefig_path + "Classification_and_Albedo_Data_{}.nc".format(area_label))

    if savefigs:
        cmap1 = mpl.colors.ListedColormap(
            ['purple', 'white', 'royalblue', 'black', 'lightskyblue', 'mediumseagreen', 'darkgreen'])
        cmap1.set_under(color='white')  # make sure background is white
        cmap2 = plt.get_cmap('Greys_r')  # reverse greyscale for albedo
        cmap2.set_under(color='white')  # make sure background is white
        ytickNames = [str(dataset.latitude[0,0].values),str(dataset.latitude[0,2750].values),str(dataset.latitude[-1,-1].values)]
        xtickNames = [str(dataset.longitude[0,0].values),str(dataset.longitude[0,2750].values),str(dataset.longitude[-1,-1].values)]
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
        plt.xticks([0, 2745, 5490],xtickNames,fontsize=26, rotation=45),plt.xlabel('Longitude (decimal degrees)',fontsize=26)
        plt.yticks([0,2745, 5490],ytickNames,fontsize=26),plt.ylabel('Latitude (decimal degrees)',fontsize=26)
        plt.grid(None)

        # second subplot = albedo map
        ax2 = plt.subplot(212)
        img2 = plt.imshow(albedo, cmap=cmap2, vmin=0, vmax=1)
        cbar2 = plt.colorbar(mappable=img2, fraction=0.045)
        cbar2.ax.set_yticklabels(labels=[0,0.2,0.4,0.6,0.8,1.0], fontsize=26)
        cbar2.set_label('Albedo',fontsize=26)
        plt.xticks([0, 2745, 5490],xtickNames,fontsize=26, rotation=45),plt.xlabel('Longitude (decimal degrees)',fontsize=26)
        plt.yticks([0,2745, 5490],ytickNames,fontsize=26),plt.ylabel('Latitude (decimal degrees)',fontsize=26)
        plt.grid(None),plt.title("Albedo Map\nProjection: UTM Zone 23, 1 pixel = 20 x 20 m",fontsize=30)
        ax2.set_aspect('equal')
        plt.tight_layout()

        plt.savefig(str(savefig_path + "Sentinel_Classified_Albedo_{}.png".format(area_label)), dpi=300)

    return predicted, albedo, dataset

def albedo_report(predicted, albedo, masterDF, merge_tile_albedo = True, save_albedo_data = False):

    """
    function calculates albedo and classification statistics and saves to file

    """
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


    if merge_tile_albedo:
        masterDF = masterDF.append(albedoDF, ignore_index=True)
        if save_albedo_data:
            masterDF.to_csv(savefig_path+'Raw_albedo_data_MERGED.csv')
            masterDF.groupby(['pred']).count().to_csv(savefig_path+'Surface_Type_Counts_MERGED.csv')
            masterDF.groupby(['pred']).describe()['albedo'].to_csv(savefig_path+'Albedo_summary_stats_MERGED.csv')

    elif save_albedo_data:
        albedoDF.to_csv(savefig_path + 'RawAlbedoData_{}.csv'.format(area_label))
        albedoDF.groupby(['pred']).count().to_csv(savefig_path+'Surface_Type_Counts_{}.csv'.format(area_label))
        albedoDF.groupby(['pred']).describe()['albedo'].to_csv(savefig_path+'Albedo_summary_stats_{}.csv'.format(area_label))

    # report summary stats
    print('\n INDIVIDUAL TILE STATS:')
    print('\n {}'.format(area_label))
    print('\n Surface type counts: \n', albedoDF.groupby(['pred']).count())
    print('\n Summary Statistics for ALBEDO of each surface type: \n',albedoDF.groupby(['pred']).describe()['albedo'])

    print('\n "Percent coverage by surface type: \n')
    print(' HA coverage = ',np.round(HApercent,2)*100,'%\n','LA coverage = ',np.round(LApercent,2)*100,'%\n','CI coverage = ',
          np.round(CIpercent,2)*100,'%\n', 'CC coverage = ',np.round(CCpercent,2)*100,'%\n', 'SN coverage = ',
          np.round(SNpercent,2)*100,'%\n', 'WAT coverage = ', np.round(WATpercent,2)*100,'%\n', 'Total Algal Coverage = ',
          np.round(HApercent+LApercent,2)*100)

    return albedoDF





# RUN FUNCTIONS

savefig_path,img_paths, Sentinel_templates, mask_in, mask_out, area_labels, masterDF = set_paths(virtual_machine=False)

for i in np.arange(0,len(area_labels),1):

    area_label = area_labels[i]
    img_path = img_paths[i]
    Sentinel_template = Sentinel_templates[i]

    #create dataset
    S2vals, clf = load_model_and_images(img_path)

    #format mask
    mask_array = format_mask (Sentinel_template,mask_in,mask_out)

    # apply model to Sentinel2 image
    predicted, albedo, dataset = ClassifyImages(S2vals, clf, mask_array, area_label, savefigs=True, save_netcdf=True)

    # calculate spatial stats
    albedoDF = albedo_report(predicted, albedo, masterDF, merge_tile_albedo = True, save_albedo_data = True)

    print('\n NOW RUNNING: ','*** ', area_label, ' ****')

