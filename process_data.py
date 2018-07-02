from random import randint, shuffle
from utils import rgb_2_labels, labels_2_rgb, softmax
import numpy as np
from scipy.misc import imread
import cv2
import os
from PIL import Image

class DataProvider:

    labeled_idx = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]
    test_idx = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38]

    def __init__(self, data_path):
        # self.input_folder = f"{data_path}/top/"
        self.input_folder = "{0}/top/".format(data_path)
        # self.gt_folder = f"{data_path}/gts_for_participants/"
        self.gt_folder = "{0}/gts_for_participants/".format(data_path)
        self.dsm_folder = "{0}/top/".format(data_path)

    def load_data(self, images_from_each=1000, image_size=224, ground_truth=False, take_all=0):
        
        """
        take_all : 0 : return all labeled idx,
                   1 : return 80% of labeled idx
                   -1 : return 20% of labeled idx
        """
        X = []
        Y = []

        indexes = DataProvider.labeled_idx if ground_truth else DataProvider.test_idx
        if take_all == 1:
            indexes = indexes[: 12]
        if take_all == -1:
            indexes = indexes[12: ]
        for idx in indexes:
            # input_path = self.input_folder + f"top_mosaic_09cm_area{str(idx)}.tif"
            input_path = self.input_folder + \
                "top_mosaic_09cm_area{0}.tif".format(str(idx))
            if ground_truth:
                # gt_path = self.gt_folder + f"top_mosaic_09cm_area{str(idx)}.tif"
                gt_path = self.gt_folder + \
                    "top_mosaic_09cm_area{0}.tif".format(str(idx))

            input_image = imread(input_path, mode='RGB')
            if ground_truth:
                gt_image = rgb_2_labels(imread(gt_path, mode='RGB'))
                print gt_image.shape

            x_dim, y_dim, _ = input_image.shape

            for _ in xrange(images_from_each):
                u = randint(0, x_dim - image_size - 1)
                v = randint(0, y_dim - image_size - 1)
                input_image[u:(u + image_size), v:(v + image_size), :]
                X.append(input_image[u:(u + image_size),
                                     v:(v + image_size), :])
                if ground_truth:
                    Y.append(gt_image[u:(u + image_size),
                                      v:(v + image_size), :])

        X = np.asarray(X)
        if ground_truth == False:
            return X
        Y = np.asarray(Y)
        print X.shape, Y.shape
        perm = np.arange(X.shape[0])
        np.random.shuffle(perm)
        X = X[perm]
        Y = Y[perm]
        test_size = int(X.shape[0] * 0.8)
        #test_data = BatchDataset(X[:test_size], Y[:test_size])
        #valid_data = BatchDataset(X[test_size:], Y[test_size:])

        return X[:test_size], Y[:test_size], X[test_size:], Y[test_size:]

    def get_dsm_data(self, image_idx):
        """
        Return numpy array contain the dsm data, size [width, height, 1]
        """
        input_path = self.dsm_folder + \
            "dsm_mosaic_09cm_area{0}.tif".format(str(image_idx))
        im = Image.open(input_path)
        w, h = im.shape
        return np.reshape(im, (w, h, 1))

    def get_full_resolution_data(self, image_idx):
        """
        Return numpy array of full size image
        """
        input_path = self.input_folder + \
            "top_mosaic_09cm_area{0}.tif".format(str(image_idx))
        input_image = imread(input_path)

        return np.asarray(input_image)

    def get_chunk_data(self, image_idx, chunk_size=224, overlap_size=112):
        """
        Split big image into multiple overlapping square chunk for prediction
        This function return a numpy array of chunks and a list of chunks position
        """
        input_path = self.input_folder + \
            "top_mosaic_09cm_area{0}.tif".format(str(image_idx))
        input_image = imread(input_path)
        x_dim, y_dim, _ = input_image.shape
        print x_dim, y_dim
        chunks = []
        chunks_info = []
        for i in xrange(0, x_dim, chunk_size - overlap_size):
            for j in xrange(0, y_dim, chunk_size - overlap_size):
                u, v = i, j
                if i + chunk_size > x_dim:
                    u = x_dim - chunk_size
                if j + chunk_size > y_dim:
                    v = y_dim - chunk_size
                chunks.append(
                    input_image[u:u + chunk_size, v:v + chunk_size, :])
                chunks_info.append((u, v))

        return np.array(chunks), chunks_info

    def save_full_resolution_image(self, image_idx, class_map):
        height, width = class_map.shape
        class_map = np.reshape(class_map, (height, width, 1))
        result = labels_2_rgb(class_map)
        im = Image.fromarray(result.astype('uint8'))
        im.save("top_mosaic_09cm_area{0}_class.tiff".format(str(image_idx)))

    def merge_chunks(self, image_idx, chunks_prediction, chunks_info, return_softmax, is_rgb=True, chunk_size=224, overlap_size=112):
        """
        Merge chunks prediction results into big image and save it:
            return_softmax: return value is softmax probabilities array if softmax is True,
                        return label image otherwise
        """
        # Fix merge_chunks
        input_path = self.input_folder + \
            "top_mosaic_09cm_area{0}.tif".format(str(image_idx))
        img = cv2.imread(input_path)
        height, width, _ = img.shape
        heat_map = np.zeros(shape=(height, width, 6))

        for idx in xrange(chunks_prediction.shape[0]):
            u, v = chunks_info[idx]
            heat_map[u:u + chunk_size, v:v +
                     chunk_size, :] += chunks_prediction[idx]

        if return_softmax:
            return softmax(heat_map)
        class_map = np.argmax(heat_map, axis=2)
        class_map = np.reshape(class_map, (height, width, 1))
        result = labels_2_rgb(class_map)

        # if is_rgb:
        #    r, g, b = cv2.split(result)
        #    result = cv2.merge([b, g, r])
        #r, g, b = result
        # result = np.
        im = Image.fromarray(result.astype('uint8'))
        im.save("top_mosaic_09cm_area{0}_class.tiff".format(str(image_idx)))
        #cv2.imwrite("top_mosaic_09cm_area{0}_class.png".format(str(image_idx)), result, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #os.system("mv top_mosaic_09cm_area{0}_class.png top_mosaic_09cm_area{1}_class.tif".format(str(image_idx), str(image_idx)))

    def gen_results_images(self, is_rgb=True):
        """
        Loop through unlabeled data and generate corresponding results
        """
        for idx in DataProvider.test_idx:
            print idx
            chunks, chunks_info = self.get_chunk_data(idx)
            self.merge_chunks(idx, chunks, chunks_info, is_rgb)


def gen_batches(input_images, ground_truth, batch_size=128, shuffle=True):
    """
    Batches generator
    Example of usage:
    For X, Y in gen_batches(X, Y, batch_size=200, shuffle=True):
        Train with a batch
    A for loop is an epooch
    """
    if shuffle:
        idx = np.random.permutation(len(input_images))
    else:
        idx = np.arange(len(input_images))
    for start_idx in xrange(0, len(input_images) - batch_size + 1, batch_size):
        ii = idx[start_idx:start_idx + batch_size]
        yield input_images[ii], ground_truth[ii]


def gen_random_batches(input_images, ground_truth, batch_size):
    idx = np.random.randint(0, input_images.shape[0], batch_size)
    return input_images[idx], ground_truth[idx]


class BatchDataset(object):

    def __init__(self, X, Y):
        self.batch_offset = 0
        self.data_size = 0
        self.X = np.array(X)
        self.Y = np.array(Y)
        # print self.X, self.Y

    def get_next_batch(self, batch_size):
        start = self.batch_offset
        if start + batch_size > self.data_size:
            start = 0
            perm = np.arange(self.data_size)
            np.random.shuffle(perm)

            self.X = self.X[perm]
            self.Y = self.Y[perm]

            # print self.X, self.Y
        self.batch_offset = start + batch_size
        print start
        end = self.batch_offset
        print end
        print self.X[start: end], self.Y[start: end]
        return self.X[start: end], self.Y[start: end]

    def get_random_batch(self, batch_size):
        idx = np.random.randint(0, self.data_size, batch_size)
        return self.X[idx], self.Y[idx]

    def get_size(self):
        return self.X.shape[0]


if __name__ == '__main__':
    dp = DataProvider("./ISPRS_semantic_labeling_Vaihingen")
    # Get data set for prediction
    # X, _ = dp.load_data(ground_truth=False)
    # Get data set with ground truth, X and Y is list of numpy arrays
    test_data, valid_data = dp.load_data(images_from_each=5, ground_truth=True)
    print test_data.get_size(), valid_data.get_size()
    x, y = test_data.get_next_batch(2)
    print x.shape, y.shape
    print x, y
    #cv2.imshow('raw01', X[47999])
    #cv2.imshow('test01', Y[47999])
    #cv2.imshow('test02', Y[0])
    #cv2.imshow('raw02', X[0])
    # cv2.waitKey(0)
    #cv2.imwrite("result_{0}.png".format(str(image_idx)), result, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #os.system("mv result_{0}.png result_{1}.tif".format(str(image_idx), str(image_idx)))
