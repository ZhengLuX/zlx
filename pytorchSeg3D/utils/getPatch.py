import numpy as np
import random
from skimage.morphology import skeletonize

def Z_Score_normalization(image_array):
    return (image_array-np.mean(image_array)) / (np.std(image_array) + 1e-8)

def get_patch(image, label, patch_size):

    flag_label = label
    shape = flag_label.shape
    shape = np.array(shape)
    label_patch_list = []
    data_patch_list = []

    flag_record = skeletonize(flag_label.astype(np.uint8))
    i, j, k = np.where(flag_record == 1)
    loc_record = []

    np,random.seed(0)

    # 沿着中心线裁剪
    pos = 0
    neg = 0
    for index in range(i.shape[0]):
        if flag_record[i[index], j[index], k[index]] == 1:
            c = np.array([i[index], j[index], k[index]])
            s = c - patch_size // 2
            e = c + patch_size // 2
            z_s = np.zeros_like(s)

            e[s < z_s] = patch_size
            s[s < z_s] = 0

            s[e > shape] = shape[e > shape] - patch_size
            e[e > shape] = shape[e > shape]

            label_patch = label[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
            label_patch_list.append(label_patch)

            data_patch = image[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
            data_patch_list.append(data_patch)

            loc_record.append(s)
            flag_record[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = 0
            pos = pos + 1
    #
    # x = np.arange(patch_size // 2, shape[0], patch_size)
    # y = np.arange(patch_size // 2, shape[1], patch_size)
    # z = np.arange(patch_size // 2, shape[2], patch_size)
    # i, j, k = np.meshgrid(x, y, z)
    # i, j, k = i.flatten(), j.flatten(), k.flatten()
    # arr = np.arange(i.shape[0])
    # np.random.shuffle(arr)
    # arr = arr.tolist()
    #
    # for index in arr:
    #     c = np.array([i[index], j[index], k[index]])
    #     s = c - patch_size // 2
    #     e = c + patch_size // 2
    #     z_s = np.zeros_like(s)
    #
    #     e[s < z_s] = patch_size
    #     s[s < z_s] = 0
    #
    #     s[e > shape] = shape[e > shape] - patch_size
    #     e[e > shape] = shape[e > shape]
    #
    #     label_patch = label[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
    #
    #     if label_patch.sum() == 0:
    #         label_patch_list.append(label_patch)
    #         data_patch = image[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
    #         data_patch_list.append(data_patch)
    #
    #         loc_record.append(s)
    #         neg = neg + 1
    #
    #     if neg >= pos:
    #         break

    return data_patch_list, label_patch_list, loc_record


def get_patch_valid(image, label, patch_size):

    shape = label.shape
    shape = np.array(shape)
    label_patch_list = []
    data_patch_list = []

    img_record = label.copy()
    # i,j,k=np.where(label==1)
    x = np.arange(patch_size // 2, shape[0], patch_size // 2)
    y = np.arange(patch_size // 2, shape[1], patch_size //2)
    z = np.arange(patch_size // 2, shape[2], patch_size //2)
    i, j, k = np.meshgrid(x, y, z)
    i, j, k = i.flatten(), j.flatten(), k.flatten()

    loc_record = []

    for index in range(i.shape[0]):
        c = np.array([i[index], j[index], k[index]])
        s = c - patch_size // 2
        e = c + patch_size // 2
        z_s = np.zeros_like(s)

        e[s < z_s] = patch_size
        s[s < z_s] = 0

        s[e > shape] = shape[e > shape] - patch_size
        e[e > shape] = shape[e > shape]

        label_patch = label[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
        label_patch_list.append(label_patch)

        data_patch = image[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
        data_patch_list.append(data_patch)

        loc_record.append(s)

        img_record[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = 0

    return data_patch_list, label_patch_list, loc_record