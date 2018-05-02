# Gaussian_Processes_Sampling
This codebase can take training images, then given a map and sample location-image pairs will give probability of environment in each location.


An example of how our system can be run is shown in example.py.

Simply run example.py and two treads (acting as two different clients) (one image classification and one adaptive sampling) will make call to our GaussianProcess module and the output is visualized in realtime using maplotlib.