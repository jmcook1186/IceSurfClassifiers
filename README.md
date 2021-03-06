# IceSurfClassifiers

This repository contains code for automated classification and estimation of albedo for ice surfaces using scikit-learn classificatiers trained using field spectroscopy and applied to multispectral images from UAVs and satellites.

The training set comprises individual reflectance spectra obtained in summer 2016 and 2017 on the Greenland Ice Sheet, ca. 38km inland of the margin at Russell Glacier, near Kangerlussuaq as part of the National Environmental Research Council's Black and Bloom project. The spectra were obtained at solar noon +/- 2 hours under constant cloud conditions using an ASD field spec pro 3 with a fibre optic collimated with a 10 degree lens levelled on a horizontal boom so that the sensor was pointd downwards, 50 cm over the sample surface. The measurements are all relative to a flat, clean Spectralon panel. Each sample was qualitatively assigned to a category (high algal biomass, low algal biomass, clean ice, snow, water, cryoconite) based on a visual inspection of the surface. For sites in a designated "sacrifical zone", immediately after the spectral reflectance data was collected, the sample surface was chipped into a sterile sampling bag and returned to the laboratory for impurity analysis. 

Note that due to the large file sizes, the input images are not provided in this repository. Suitable Sentinel-2 images are freely available from various online sourcs including sentinelhub and earthexplorer.usgs.gov. Advice on downloading the relevant files and organising into correct folder structures is provided in the script annotations.


### UAV Imagery ### 

The 200 x 200m area immediately adjacent to and including the "sacrificial zone" was then imaged using a Micasense Red-Edge camera gimbal-mounted onto a custom quadcopter. The retrieved images had a ground resolution of 5cm and were radiometrically calibrated according to micasense guidance. A Micasense reflectance panel and a Spectralon standard were visible in every image.

Before the field spectra were used as training data, the dataset was reduced to only those wavelengths that matched those of the drone camera. The reflectance at these key wavelengths were used as the dataset features, while the surface class was used as the label.

The algorithms provided in this repository train a set of supervised classifiers (KNN, Naive Bayes, SVM, voting ensemble and Random Forest) on the field spectra using cross validation. A subset is withheld for testing the model performance.

Once trained, the model is applied to the drone imagery, with the reflectance at each wavelength providing features leading to an estimate of the surface type for each individual pixel.

For the UAV images there is an additional option to extend the training data by selecting areas in UAV images wthat are homogenous and the surfce classification known with certainty, e.g. clearly identifiable snow packs or ponds.

In addition to classification, the script calculates the surface albedo pixelwise from the reflectance of each spectral band using a narrowband to broadband conversion (Knap et al. 1999). Spatial statistics are reported to the console and saved to csv.


### Prerequisites ###

These scripts are written in Python 3.5 and require several packages with co-dependencies. Suggest the following fresh environment configuration:

conda create --name IceSurfClassifiers python=3.5 matplotlib scikit-learn gdal rasterio numpy pandas seaborn

then conda install -c conda-forge xarray georaster, sklearn_xarray

There is also preprocessing required for the UAV and Sentinel imagery. The UAV imagery requires stitching, georeferencing and radiometric calibration folowing the Micasense protocols.
Sentinel-2 imagery requires atmospheric correction and consistent ground resolution achieved usi Sen2Cor, plus downloading of the appropriate mask from the Greenland Ice Mapping Project.
Detailed instructions are provided in the script annotations.



### Sentinel 2 Imagery ###

The same process was followed for supervised classification of Sentinel-2 satellite remote sensing data covering a much larger area around the field site. These were downloaded from SentinelHub. Sentinel 2 has greater spectral resolution that the rededge camera, so the model is trained on 9 wavelengths rather than 5. 

Further details are available in the code annotations.


For Sentinel 2 image data, the classified map and albedo map are masked using the Greenland Ice Mapping Project land/ice mask to exclude non-ice areas from the spatial analysis. Albedo data is collated for all tiles and reported to the console and saved to csv as a single summary dataset. Albedo data for individual tiles is an optional output.


### Example Outputs ###

The figures below show example outputs from the UAV and Sentinel-2 classifiers respectively. The UAV image was collected over a 200 x 200m area on the Greenland Ice Sheet surface in 2017. The Sentinel 2 image is for a tile North of the UAV area. The Sentinel-2 code provided here outputs a similar plot for all requested tiles.


![Example output plot from UAV classifiers code](./UAV_Classifier_Example_Plot.png?raw=true/width=10 "Example output plot from UAV classifiers code")

![Example output plot from Sentinel classifiers code](./Sentinel_Classifier_Example_Plot.png?raw=true/width=10 "Example output plot from Sentinel classifiers code")

### Notes on Classification Method ###

In this project we have trained a random forest classifier on spectral reflectance measurements made on the ground using a field spectrometer. The hyperspectral data was reduced down to those wavelengths that are also measured by the remote cameras, either on our UAV or on Sentinel-2. This approach is quite novel and it has benefits and drawbacks. The main drawback is that the data is expensive and time consuming to collect, as it has to be collected in person on the ice surface, limiting the size of the available training data. However, the benefit is that the training data is of the highest quality. This is because we can be completely confident in the labels, since for each spectrum the sample area is homogenous and the ice surface has been scraped away for analysis in the laboratory. We are not vulnerable to ambiguity introduced by sub-pixel heterogeneity or uncertainty in subjective visual assessments of the surface. For the classifier itself, we tested a suite of algorithms and simply chose the one that performed best on our test set. 

### Permissions ###

This code is provided without warranty or guarantee of any kind. Usage should cite the doi for v1.0 release of this repository (doi:10.5281/zenodo.2598122) and the paper:

Cook, J. M., Tedstone, A. J., Williamson, C., McCutcheon, J., Hodson, A. J., Dayal, A., Skiles, M., Hofer, S., Bryant, R., McAree, O., McGonigle, A., Ryan, J., Anesio, A. M., Irvine-Fynn, T. D. L., Hubbard, A., Hanna, E., Flanner, M., Mayanna, S., Benning, L. G., van As, D., Yallop, M., McQuaid, J., Gribbin, T., and Tranter, M.: Glacier algae accelerate melt rates on the western Greenland Ice Sheet, The Cryosphere Discuss., <https://doi.org/10.5194/tc-2019-58>, in review, 2019.
