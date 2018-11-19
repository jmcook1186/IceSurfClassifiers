# IceSurfClassifiers

This repositorey contains code in active development for automated classification of ice surfaces using machine learning algorithms trained using ground-based spectroscopy and applied to multispectral images from UAVs and satellites.

The training set comprises individual reflectance spectra obtained in summer 2016 and 2017 on the Greenland Ice Sheet, ca. 38km inland of the margin at Russell Glacier, near Kangerlussuaq as part of the National Environmental Research Council's Black and Bloom project. The spectra were obtained at solar noon +/- 2 hours under constant cloud conditions using an ASD field spec pro 3 with a fibre optic collimated with a 10degree lens levelled on a horizontal boom so that the sensor was pointd downwards, 50 cm over the sample surface. The measurements are all relative to a flat, clean Spectralon panel. Each sample was qualitatively assigned to a category (high algal biomass, low algal biomass, clean ice, snow, water, cryoconite) based on a visual inspection of the surface. For sites in a designated "sacrifical zone", immediately after the spectral reflectance data was collected, the sample surface was chipped into a sterile sampling bag and returned to the laboratory for impurity analysis. 

The 200 x 200m area immediately adjacent to and including the "sacrificial zone" was then imaged using a Micasense Red-Edge camera gimbal-mounted onto a custom quadcopter. The retrieved images had a ground resolution of 5cm and were radiometrically calibrated according to micasense guidance. A Micasense reflectance panel and a Spectralon standard were visible in every image.

Before the field spectra were used as training data, the dataset was reduced to only those wavelengths that matched those of the drone camera. The reflectance at these key wavelengths were used as the dataset features, while the surface class was used as the label.

The algorithms provided in this repository train a set of supervised classifiers (KNN, Naive Bayes, SVM, voting ensemble and Random Forest) on the field spectra using cross validation. A subset is withheld for testing the model performance.

Once trained, the model is applied to the drone imagery, with the reflectance at each wavelength providing features leading to an estimate of the surface type for each individual pixel.

The same process was followed for supervised classification of Sentinel-2 satellite remote sensing data covering a much larger area around the field site. These were downloaded from earth-explorer.usgs.gov. Sentinel 2 has slightly bwtter spectral resolution that the rededge camera, so the model is trained on 9 wavelengths rather than 5. 

Further details are available in the code annotations.


# Permissions

There are no permissions associated with this code, it is unpublished and still under development. please do not use without explicit permission from the author. 

