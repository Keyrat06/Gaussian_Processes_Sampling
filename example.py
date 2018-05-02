import threading
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import numpy as np
import cv2
import GaussianProcess
import util

plt.ion()

N = util.N
M = util.M
K = 2

MAP = (np.ones((N, M, K))/K)

def Image_Classification_Thread():
    print("YAY Image Classification Has Started!")
    GaussianProcess.setup()
    MAP = cv2.imread("MAP.png")
    FEATURE_EXTRACTOR = lambda image: [image[:, :, 0].mean(), image[:, :, 1].mean(), image[:, :, 2].mean()]
    i = 0
    while True:
        sample_location = (np.random.randint(0, N), np.random.randint(0, M))
        image_sample = MAP[sample_location[0]*100:sample_location[0]*100+100,
                           sample_location[1]*100:sample_location[1]*100+100]

        image_feature = FEATURE_EXTRACTOR(image_sample)
        time.sleep(0.2)
        GaussianProcess.new_image(image_feature, sample_location[0], sample_location[1])
        i+=1

def Adaptive_Sampling_Thread():
    print("YAY Adaptive Sampling Has Started!")
    while True:
        time.sleep(1)
        global MAP
        MAP = GaussianProcess.get_image_map()


def main():
    image = threading.Thread(name='image_class', target=Image_Classification_Thread)
    sampling = threading.Thread(name='adaptive_sample', target=Adaptive_Sampling_Thread)
    image.start()
    sampling.start()




    while True:
        plt.clf()
        cm = plt.imshow(MAP[:, :, 0].T, vmin=0, vmax=1)
        plt.colorbar(cm)
        plt.pause(1)


if __name__ == "__main__":
    main()
