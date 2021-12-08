import os
import cv2
import itertools
import numpy as np
import pickle


class IO:
    instace = None

    @staticmethod
    def getInstance():
        if not IO.instace:
            IO.instace = IO()

        return IO.instace


    def load_data(self, path):
        images, labels = [], []

        for folder in os.listdir(path):
            for file in os.listdir(os.path.join(path, folder)):
                file = os.path.join(path, folder, file)
                image = cv2.imread(file)
                image = cv2.resize(image, (512, 328))
                images.append(image)
                labels.append(folder)

        return images, labels


    def save_model(self, path, model):
        with open(path, "wb+") as f:
            pickle.dump(model, f)


    def load_model(self, path):
        with open(path, "rb") as f:
            model = pickle.load(f)

        return model



class LBPH:
    def __init__(self, image, radius=1, gridX=8, gridY=8) -> None:
        self.image = image
        self.radius = radius
        self.gridX = gridX
        self.gridY = gridY


    def __LPB(self):
        h, w = self.image.shape[:2]
        ks = (2 * self.radius + 1) ** 2

        x, y = np.meshgrid(range(self.radius, w-self.radius), range(self.radius, h-self.radius))
        slice_x = np.expand_dims(x, axis=-1) * np.ones(ks) + np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        slice_y = np.expand_dims(y, axis=-1) * np.ones(ks) + np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        slice_x = slice_x.reshape((-1, ks)).astype('int32')
        slice_y = slice_y.reshape((-1, ks)).astype('int32')

        # thresholding pixel
        thresholds = self.image[y, x].reshape((-1, 1))
        binary = np.zeros(slice_x.shape[:2])
        binary[self.image[slice_y, slice_x] >= thresholds] = 1

        # concate binary
        cols = list(range(ks))[:ks//2] + list(range(ks))[ks//2+1:]
        binary = binary[:, cols]

        # convert binary to decimal
        coef = np.arange(ks-2, -1, -1)
        decimals = np.sum(binary * np.power(2, coef), axis=1).reshape((x.shape[:2]))

        # label pixel with decimal above
        image = np.copy(self.image)
        image[y, x] = decimals

        return image


    def __H(self, image):
        # divine image into grid
        blocks = np.array_split(image, self.gridX, axis=1)
        blocks = list(map(lambda block: np.array_split(block, self.gridY), blocks))

        # concate grid and extract histogram
        blocks = list(itertools.chain(*blocks))
        his = np.array([np.bincount(block.flatten(), minlength=256) / (block.shape[0]*block.shape[1]) for block in blocks])

        return his.flatten()

    
    def getDistribution(self):
        decimal = self.__LPB()
        distribution = self.__H(decimal)
        
        return distribution



def convertChannel(image, mode):
    channel = None

    if mode == "BGR2RGB":
        channel = cv2.COLOR_BGR2RGB
    elif mode == "BGR2GRAY":
        channel = cv2.COLOR_BGR2GRAY
    elif mode == "RGB2GRAY":
        channel = cv2.COLOR_RGB2GRAY
    elif mode == "GRAY2RGB":
        channel = cv2.COLOR_GRAY2RGB
    elif mode == "GRAY2RGB":
        channel = cv2.COLOR_GRAY2RGB

    return cv2.cvtColor(image, channel)


# Load Viola-Jones pretrained model
face_classifier = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")