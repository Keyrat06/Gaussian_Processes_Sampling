import threading
import matplotlib.pyplot as plt
import pickle
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
np.random.seed(0)

def Image_Classification_Thread(n = float('inf')):
    print("YAY Image Classification Has Started!")
    GaussianProcess.setup()
    imageClassifier = pickle.load(open("Image_Classifier_Model.p", "rb"))
    MAP = cv2.imread("MAP.png")
    FEATURE_EXTRACTOR = lambda image: [image[:, :, 0].mean(), image[:, :, 1].mean(), image[:, :, 2].mean()]
    i = 0
    while True and i < n:
        sample_location = (np.random.randint(0, N), np.random.randint(0, M))
        image_sample = MAP[sample_location[0]*100:sample_location[0]*100+100,
                           sample_location[1]*100:sample_location[1]*100+100]

        image_feature = FEATURE_EXTRACTOR(image_sample)
        # time.sleep(0.01)
        P = imageClassifier.predict_proba(np.array([image_feature]))[0]
        GaussianProcess.new_image(P, sample_location[0], sample_location[1])
        i += 1



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
        print(util.get_NLL(MAP))
        plt.clf()
        cm = plt.imshow(MAP[:, :, 1], vmin=0, vmax=1)
        plt.colorbar(cm)
        plt.pause(1)


def experament(a_options=np.linspace(0,1,11), b_options=range(1,21), n=100):
    """
    This function just finds optimal a and b values and plots the space
    :param a_options: list options for a
    :param b_options: list options for b
    :param n: number of samples
    :return: None
    """
    np.random.seed(0)
    Image_Classification_Thread(n)
    data = np.zeros((len(a_options), len(b_options)))
    min_NLL = float('inf')
    optimal_params = (-1, -1)
    for i, a in enumerate(a_options):
        for j, b in enumerate(b_options):
            MAP = GaussianProcess.get_image_map(a, b)
            nll = util.get_NLL(MAP)
            data[i, j] = nll
            if nll < min_NLL:
                optimal_params = (a, b)
                min_NLL = nll
    print("optimal a = {}, optimal b = {}".format(*optimal_params))
    plt.imshow(data)
    plt.xticks(range(20), b_options)
    plt.yticks(range(10), a_options)
    plt.xlabel("b")
    plt.ylabel("a")
    plt.show("hold")

experament()

if __name__ == "__main__":
    main()



