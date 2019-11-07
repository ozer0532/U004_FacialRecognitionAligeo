import cv2
import numpy as np
import scipy
from scipy import spatial
from imageio import imread
import _pickle as pickle
import random
import os
import matplotlib.pyplot as plt
import math

# Feature extractor
def extract_features(image_path, vector_size = 32):
    image = imread(image_path)
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.AKAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print ('Error: ', e)
        return None

    return dsc



class Matcher(object):

    def __init__(self, pickled_db_path="features.pck"):
        with open(pickled_db_path, 'rb') as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

    def cos_cdist(self, vector):
        # getting cosine distance between search image and images database
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

    def match(self, image_path, topn=5):
        features = extract_features(image_path)
        img_distances = self.cos_cdist(features)
        # getting top 5 records
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()
        return nearest_img_paths, img_distances[nearest_ids].tolist()

    def euclidean(self,image_path, topn=5):
        features = extract_features(image_path)
        sets = self.matrix
        distance = []
        for i in sets:
            sum = 0
            for j in range(len(features)):
                sum += (i[j]-features[j]) ** 2
            d = sum ** (1/2)
            distance.append(d)
        img_distances = np.array(distance)

        # getting top 5 records
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()
        return nearest_img_paths, img_distances[nearest_ids].tolist()

    def cos_sim(self,image_path, topn=5):
        features = extract_features(image_path)
        sets = self.matrix
        distance = []
        for i in sets:
            sum = 0
            sum1 = 0
            sum2 = 0
            for j in range(len(features)):
                sum += (i[j]*features[j])
                sum1 += (i[j]) ** 2
                sum2 += (features[j]) ** 2
            scaV = sum1 ** (1/2)
            scaW = sum2 ** (1/2)
            d = (sum)/(scaV * scaW)
            distance.append(d)
        img_distances = np.array(distance)
        
        # getting top 5 records
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()
        return nearest_img_paths, img_distances[nearest_ids].tolist()



def show_img(path):
    img = imread(path, pilmode="RGB")
    plt.imshow(img)
    plt.show()
"""    
def run():
    dataset_path = 'dataset/'
    test_path = 'testset/'
    dataset_files = [os.path.join(dataset_path, p) for p in sorted(os.listdir(dataset_path))]
    test_files = [os.path.join(test_path, p) for p in sorted(os.listdir(test_path))]
    # getting 3 random images 
    sample = random.sample(test_files, 1)
    action = input();
    if (action == "input"):
        batch_extractor(dataset_path)
    elif (action == "test"):
        ma = Matcher('features.pck')
        
        for s in sample:
            print ('Query image ==========================================')
            show_img(s)
            names, match = ma.match(s, topn=3)
            print ('Result images ========================================')
            for i in range(3):
                # we got cosine distance, less cosine distance between vectors
                # more they similar, thus we subtruct it from 1 to get match value
                print ('Match %s' % (match[i]))
                show_img(os.path.join(dataset_path, names[i]))
"""
#run()