# Dehazing for Microscopy Data
You can find the code for my Bachelor thesis in this repository. Here I want to assist you with a little tutorial-like 
guideline.

## Prerequisites
* Python 3.5+ installed
* Repository cloned
* Install all Libraries which are used for each method via pip install
* Import all methods into your new file by running:
````
import methods.*
````

## Data Generator
To generate your own synthetic data you can use the following code and tune the parameters to get the data
as you want. 

```
data = DataGenerator.ImagePair(seed=42, amount=15)
data.gen_multiple_images(image_shape=(256, 256), min_angle=35, max_angle=100,
                         line_width=2, curve_count=15, clean_curves=4, fadeout=0.5, weight=0.1
                         intensity=100, overlapping=True)
```
The code above creates an instance of ImagePair with 15 image pairs.
```   
data.offset(30)
data.set_poisson_noise(1.5)
data.set_gauss_noise(20, 13)
data.set_fractal_noise(0, 60, (8, 8), 2, 0.5, lazy=True)
```
In this snippet of code we added some noise to the images.
``` 
data.plot([0, 5, 9])
```
Calling this function plots the image pairs 0, 5 and 9.
```
data.save(path='./new_dir', name='data')
```
This code saves the ground truth and degraded images as 'ground_truth_images_data.tif' and 
'degraded_images_data.tif'. Keep in mind, that the Data Generator returns you the generated images as '.tif'-files.

## Dehazing with Multiscale Wavelet Decomposition
To dehaze an image it is important to have your data saved as '.tif'-file. Fist, it is
important to choose a wavelet for the multiscale decomposition. You can display each
available wavelet by running the following command of the python wavelet analysis library.
````
pywt.wavelist()
````
The following code uses your tif-image with the specific slide in the tif as well as 
the daubechies2 wavelet to dehaze an image at the 5th decomposition level. The threshold parameter is necessary 
to avoid wavelet-decomposition-caused artifacts.

```
dehazed_image = DMWD.WaveletDehaze('db2')
dehazed_image.dehaze(image='YourTiff.tif', slide=0, level=5, threshold=0)
```
After dehazing your image you can return it by running the following command.
```
dehazed_image.return_image_dehazed()
```
Now, you can use it for further processing.

## Top Hat Transforms

To dehaze images via Top Hat Transforms you can use the code in the TopHat.py file as a reference. Therefore the 
following code describes the case when you just want to run the Top Hat Transform on only one window size and one
threshold parameter. 
```
top_hat_transformed_image = TopHat.tophat(image, window_size=1, threshold=0)
```
You have only to replace 'image' with your image (2d-array, no '.tif'-file) which should be dehazed, insert your 
window_size and also your threshold. Done.



