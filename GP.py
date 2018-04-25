import sklearn.gaussian_process as gp
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob


FEATURE_EXTRACTOR = lambda image: [image[:, :, 0].mean(), image[:, :, 1].mean(), image[:, :, 2].mean(), image.mean()]

auditorium_x = np.vstack([FEATURE_EXTRACTOR(cv2.imread(x)) for x in glob("Auditorium/*.jpg")])
bowling_x = np.vstack([FEATURE_EXTRACTOR(cv2.imread(x)) for x in glob("Bowling/*.jpg")])

y = np.hstack((np.ones(len(auditorium_x)), 0*np.ones(len(bowling_x))))
auditorium_x = np.vstack((auditorium_x, bowling_x))

GPC_images_classifier = gp.GaussianProcessClassifier(1.0 * gp.kernels.RBF([1.0, 2.0, 2.0, 1.0]))
GPC_images_classifier.fit(auditorium_x, y)

MAP = cv2.imread("Example_Map.png")
samples = np.array([[0, 0], [1, 1], [1, 3], [2, 4], [2, 3], [2, 2], [2, 4]])
image_samples = np.array([FEATURE_EXTRACTOR(MAP[x[0]*100:x[0]*100+100, x[1]*100:x[1]*100+100]) for x in samples])
Y = GPC_images_classifier.predict_proba(image_samples)[:, 1]

print(Y)

GPC_position_classifier = gp.GaussianProcessRegressor(1.0 * gp.kernels.RBF([1.0]) + gp.kernels.WhiteKernel())
GPC_position_classifier.fit(samples, Y)

X = np.array(np.meshgrid(np.arange(3), np.arange(5))).T.reshape((-1, 2))
probabilities = GPC_position_classifier.predict(X).reshape((3, 5))
cm = plt.imshow(probabilities)
plt.colorbar(cm)
plt.show()
