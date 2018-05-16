import threading
import matplotlib.pyplot as plt
import pickle
import time
import numpy as np
import cv2
import GaussianProcess
import util
from scipy import interpolate
plt.ion()

N = util.N
M = util.M
K = 2

# MAP = np.ones((N,M,K))/float(K)
MAP = lambda x, y: np.ones(K)/float(K)
# np.random.seed(0)

def Image_Classification_Thread(n = float('inf'), t=0.1):
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
        time.sleep(t)
        P = imageClassifier.predict_proba(np.array([image_feature]))[0]
        GaussianProcess.new_image(P, sample_location[0], sample_location[1])
        i += 1



def Adaptive_Sampling_Thread():
    print("YAY Adaptive Sampling Has Started!")
    while True:
        time.sleep(0.1)
        global MAP
        # MAP = GaussianProcess.get_image_map()
        MAP = GaussianProcess.GPRegressor()


def main():
    image = threading.Thread(name='image_class', target=Image_Classification_Thread)
    sampling = threading.Thread(name='adaptive_sample', target=Adaptive_Sampling_Thread)
    image.start()
    sampling.start()

    while True:
        plt.pause(1)
        plt.clf()
        MAP.visualize(0)
        # print(util.get_NLL(MAP))
        # plt.clf()
        # cm = plt.imshow(MAP[:, :, 1], vmin=0, vmax=1)
        # plt.colorbar(cm)
        # plt.pause(1)
        # xs = []
        # ys = []
        # ps = []
        # for _ in range(1000):
        #     plt.clf()
        #     x = np.random.random() * N
        #     y = np.random.random() * M
        #     p = MAP(x,y)
        #     xs.append(x)
        #     ys.append(y)
        #     ps.append(p[0])
        # numIndexes = 500
        # xi = np.linspace(np.min(xs), np.max(xs), numIndexes)
        # yi = np.linspace(np.min(ys), np.max(ys), numIndexes)
        # XI, YI = np.meshgrid(xi, yi)
        # points = np.vstack((xs, ys)).T
        # values = np.array(ps)
        # print(points.shape)
        # print(values.shape)
        # DEM = interpolate.griddata(points, values, (XI, YI), method='linear', fill_value=0.02)
        # print(DEM)
        # levels = np.arange(np.min(DEM), np.max(DEM), 0.1)
        # plt.contour(DEM, levels, linewidths=0.2,colors='k')
        # plt.imshow(DEM,cmap ='RdYlGn_r',origin='lower', vmin=0, vmax=1)
        # plt.colorbar()
        # xArrayNormalized = xs/(np.max(xs)-np.min(xs))
        # xArrayNormalized -= np.min(xArrayNormalized)
        # yArrayNormalized = ys/(np.max(ys)-np.min(ys))
        # yArrayNormalized -= np.min(yArrayNormalized)
        # plt.scatter(numIndexes*xArrayNormalized, numIndexes*yArrayNormalized, color='k', alpha=0.25)




def experament(a_options=np.linspace(0,1,11), b_options=range(1,21), n=100):
    """
    This function just finds optimal a and b values and plots the space
    :param a_options: list options for a
    :param b_options: list options for b
    :param n: number of samples
    :return: None
    """
    np.random.seed(0)
    Image_Classification_Thread(n, t=0)
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
    cm = plt.imshow(data)
    plt.colorbar(cm)
    plt.xticks(range(20), b_options)
    plt.yticks(range(10), a_options)
    plt.title("Negative Log Loss for values of a and b")
    plt.xlabel("b")
    plt.ylabel("a")
    plt.show("hold")

# experament()

if __name__ == "__main__":
    main()



