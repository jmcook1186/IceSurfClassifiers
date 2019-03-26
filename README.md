# IceSurfClassifiers

This repository contains code for automated classification of ice surfaces using scikit-learn classification algorithms trained using field spectroscopy and applied to multispectral images from UAVs and satellites.

The training set comprises individual reflectance spectra obtained in summer 2016 and 2017 on the Greenland Ice Sheet, ca. 38km inland of the margin at Russell Glacier, near Kangerlussuaq as part of the National Environmental Research Council's Black and Bloom project. The spectra were obtained at solar noon +/- 2 hours under constant cloud conditions using an ASD field spec pro 3 with a fibre optic collimated with a 10 degree lens levelled on a horizontal boom so that the sensor was pointd downwards, 50 cm over the sample surface. The measurements are all relative to a flat, clean Spectralon panel. Each sample was qualitatively assigned to a category (high algal biomass, low algal biomass, clean ice, snow, water, cryoconite) based on a visual inspection of the surface. For sites in a designated "sacrifical zone", immediately after the spectral reflectance data was collected, the sample surface was chipped into a sterile sampling bag and returned to the laboratory for impurity analysis. 


### UAV Imagery ### 

The 200 x 200m area immediately adjacent to and including the "sacrificial zone" was then imaged using a Micasense Red-Edge camera gimbal-mounted onto a custom quadcopter. The retrieved images had a ground resolution of 5cm and were radiometrically calibrated according to micasense guidance. A Micasense reflectance panel and a Spectralon standard were visible in every image.

Before the field spectra were used as training data, the dataset was reduced to only those wavelengths that matched those of the drone camera. The reflectance at these key wavelengths were used as the dataset features, while the surface class was used as the label.

The algorithms provided in this repository train a set of supervised classifiers (KNN, Naive Bayes, SVM, voting ensemble and Random Forest) on the field spectra using cross validation. A subset is withheld for testing the model performance.

Once trained, the model is applied to the drone imagery, with the reflectance at each wavelength providing features leading to an estimate of the surface type for each individual pixel.

For the UAV images there is an additional option to extend the training data by selecting areas in UAV images wthat are homogenous and the surfce classification known with certainty, e.g. clearly identifiable snow packs or ponds.



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


For Sentinel 2 image data, the classified map and albedo map are masked using the Greenland Ice Mapping Project land/ice mask to exclude non-ice areas from the spatial analysis.


### Example Outputs ###

The figures below show example outputs from the UAV and Sentinel-2 classifiers respectively. The UAV image was collected over a 200 x 200m area on the Greenland Ice Sheet surface in 2017. The Sentinel 2 image is for a tile North of the UAV area. The Sentinel-2 code provided here outputs a similar plot for all requested tiles.


![Example output plot from UAV classifiers code](./UAV_Classifier_Example_Plot.png?raw=true/width=50 "Example output plot from UAV classifiers code")

![Example output plot from UAV classifiers code](./master/Sentinel_Classifier_Example_Plot.png?raw=true/width=50 "Example output plot from UAV classifiers code")

### Permissions ###

The code is provided without warranty or guarantee of any kind.
