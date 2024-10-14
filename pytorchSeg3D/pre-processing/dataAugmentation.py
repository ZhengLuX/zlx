import torch
from torchvision import transforms
import SimpleITK as sitk
import numpy as np
import os
import imgaug.augmenters as iaa
import cv2

def GaussianBlur_and_save_image(image_path, output_directory, sigma):
    # 读取图像
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # 定义增强器
    aug = iaa.Sequential([
        iaa.GaussianBlur(sigma=sigma),
    ])

    # 执行图像增强
    image_augmented_list = []
    for img in image_array:
        img_augmented = aug.augment_images([img])
        image_augmented_list.append(img_augmented[0])

    # 转换为 NumPy 数组
    image_aug = np.array(image_augmented_list)

    # 创建 SimpleITK 图像对象
    blurred_image = sitk.GetImageFromArray(image_aug)

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 构建输出图像路径
    output_image_path = os.path.join(output_directory, f'GaussianBlurred{sigma}_{os.path.basename(image_path)}')

    # 保存增强后的图像
    sitk.WriteImage(blurred_image, output_image_path)

    print(f"GaussianBlurred image saved at: {output_image_path}")

def AverageBlur_and_save_image(image_path, output_directory, k):
    # 读取图像
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # 定义增强器
    aug = iaa.Sequential([
        iaa.AverageBlur(k=k),
    ])

    # 执行图像增强
    image_augmented_list = []
    for img in image_array:
        img_augmented = aug.augment_images([img])
        image_augmented_list.append(img_augmented[0])

    # 转换为 NumPy 数组
    image_aug = np.array(image_augmented_list)

    # 创建 SimpleITK 图像对象
    blurred_image = sitk.GetImageFromArray(image_aug)

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 构建输出图像路径
    output_image_path = os.path.join(output_directory, f'AverageBlurred{k}_{os.path.basename(image_path)}')

    # 保存增强后的图像
    sitk.WriteImage(blurred_image, output_image_path)

    print(f"AverageBlurred image saved at: {output_image_path}")

def MedianBlur_and_save_image(image_path, output_directory, k):
    # 读取图像
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # 定义增强器
    aug = iaa.Sequential([
        iaa.MedianBlur(k=k),
    ])

    # 执行图像增强
    image_augmented_list = []
    for img in image_array:
        img_augmented = aug.augment_images([img])
        image_augmented_list.append(img_augmented[0])

    # 转换为 NumPy 数组
    image_aug = np.array(image_augmented_list)

    # 创建 SimpleITK 图像对象
    blurred_image = sitk.GetImageFromArray(image_aug)

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 构建输出图像路径
    output_image_path = os.path.join(output_directory, f'MedianBlurred{k}_{os.path.basename(image_path)}')

    # 保存增强后的图像
    sitk.WriteImage(blurred_image, output_image_path)

    print(f"MedianBlurred image saved at: {output_image_path}")

def GaussianNoise_and_save_image(image_path, output_directory, sd):
    # 读取图像
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # 定义增强器
    aug = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=sd),
    ])

    # 执行图像增强
    image_augmented_list = []
    for img in image_array:
        img_augmented = aug.augment_images([img])
        image_augmented_list.append(img_augmented[0])

    # 转换为 NumPy 数组
    image_aug = np.array(image_augmented_list)

    # 创建 SimpleITK 图像对象
    blurred_image = sitk.GetImageFromArray(image_aug)

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 构建输出图像路径
    output_image_path = os.path.join(output_directory, f'GaussianNoise_sd{sd}_{os.path.basename(image_path)}')

    # 保存增强后的图像
    sitk.WriteImage(blurred_image, output_image_path)

    print(f"GaussianNoise image saved at: {output_image_path}")

def PoissonNoise_and_save_image(image_path, output_directory, lam):
    # 读取图像
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # 定义增强器
    aug = iaa.Sequential([
        iaa.AdditivePoissonNoise(lam=lam),
    ])

    # 执行图像增强
    image_augmented_list = []
    for img in image_array:
        img_augmented = aug.augment_images([img])
        image_augmented_list.append(img_augmented[0])

    # 转换为 NumPy 数组
    image_aug = np.array(image_augmented_list)

    # 创建 SimpleITK 图像对象
    blurred_image = sitk.GetImageFromArray(image_aug)

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 构建输出图像路径
    output_image_path = os.path.join(output_directory, f'PoissonNoise_lambda{lam}_{os.path.basename(image_path)}')

    # 保存增强后的图像
    sitk.WriteImage(blurred_image, output_image_path)

    print(f"PoissonNoise image saved at: {output_image_path}")

def SaltAndPepper_and_save_image(image_path, output_directory, p):
    # 读取图像
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # 定义增强器
    aug = iaa.Sequential([
        iaa.SaltAndPepper(p=p,seed=0)
    ])

    # 执行图像增强
    image_augmented_list = []
    for img in image_array:
        img_augmented = aug.augment_images([img])
        image_augmented_list.append(img_augmented[0])

    # 转换为 NumPy 数组
    image_aug = np.array(image_augmented_list)

    # 创建 SimpleITK 图像对象
    blurred_image = sitk.GetImageFromArray(image_aug)

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 构建输出图像路径
    output_image_path = os.path.join(output_directory, f'SaltAndPepper_p{p}_{os.path.basename(image_path)}')

    # 保存增强后的图像
    sitk.WriteImage(blurred_image, output_image_path)

    print(f"SaltAndPepper image saved at: {output_image_path}")

def LaplaceNoise_and_save_image(image_path, output_directory,sd,loc = 30):
    # 读取图像
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # 定义增强器
    aug = iaa.Sequential([
        iaa.AdditiveLaplaceNoise(loc=loc,scale=sd)
    ])

    # 执行图像增强
    image_augmented_list = []
    for img in image_array:
        # print(img)
        img_augmented = aug.augment_images([img])
        image_augmented_list.append(img_augmented[0])

    # 转换为 NumPy 数组
    image_aug = np.array(image_augmented_list)

    # 创建 SimpleITK 图像对象
    blurred_image = sitk.GetImageFromArray(image_aug)

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 构建输出图像路径
    output_image_path = os.path.join(output_directory, f'LaplaceNoise_sd{sd}_{os.path.basename(image_path)}')

    # 保存增强后的图像
    sitk.WriteImage(blurred_image, output_image_path)

    print(f"LaplaceNoise image saved at: {output_image_path}")

def MotionBlur_and_save_image(image_path, output_directory, k, angle):
    # 读取图像
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # 定义增强器
    aug = iaa.Sequential([
        iaa.MotionBlur(k=k,angle=angle),
    ])

    # 执行图像增强
    image_augmented_list = []
    # mask_augmented_list = []
    for img in image_array:
        img_augmented = aug.augment_images([img])
        image_augmented_list.append(img_augmented[0])

    # 转换为 NumPy 数组
    image_aug = np.array(image_augmented_list)

    # 创建 SimpleITK 图像对象
    blurred_image = sitk.GetImageFromArray(image_aug)

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 构建输出图像路径
    output_image_path = os.path.join(output_directory, f'MotionBlur_k{k}_angle{angle}_{os.path.basename(image_path)}')

    # 保存增强后的图像
    sitk.WriteImage(blurred_image, output_image_path)

    print(f"MotionBlur image saved at: {output_image_path}")

def SaltAndPepper_and_MedianBlur(image_path, output_directory, p, k):
    # 读取图像
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # 定义增强器
    aug = iaa.Sequential([
        iaa.SaltAndPepper(p=p),
        iaa.MedianBlur(k=k),
    ])

    # 执行图像增强
    image_augmented_list = []
    for img in image_array:
        img_augmented = aug.augment_images([img])
        image_augmented_list.append(img_augmented[0])

    # 转换为 NumPy 数组
    image_aug = np.array(image_augmented_list)

    # 创建 SimpleITK 图像对象
    sharpened_image = sitk.GetImageFromArray(image_aug)

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 构建输出图像路径
    output_image_path = os.path.join(output_directory, f'SaltAndPepper{p}_and_MedianBlur{k}_{os.path.basename(image_path)}')

    # 保存增强后的图像
    sitk.WriteImage(sharpened_image, output_image_path)

    print(f"SaltAndPepper_and_MedianBlur image saved at: {output_image_path}")

def GaussianNoise_and_GaussianBlur(image_path, output_directory, sd, sigma):
    # 读取图像
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # 定义增强器
    aug = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=sd),
        iaa.GaussianBlur(sigma=sigma),
    ])

    # 执行图像增强
    image_augmented_list = []
    for img in image_array:
        img_augmented = aug.augment_images([img])
        image_augmented_list.append(img_augmented[0])

    # 转换为 NumPy 数组
    image_aug = np.array(image_augmented_list)

    # 创建 SimpleITK 图像对象
    sharpened_image = sitk.GetImageFromArray(image_aug)

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 构建输出图像路径
    output_image_path = os.path.join(output_directory, f'GaussianNoise{sd}_and_GaussianBlur{sigma}_{os.path.basename(image_path)}')

    # 保存增强后的图像
    sitk.WriteImage(sharpened_image, output_image_path)

    print(f"GaussianNoise_and_GaussianBlur image saved at: {output_image_path}")

def Sharpen_and_save_image(image_path, output_directory):
    # 读取图像
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # 定义增强器
    aug = iaa.Sequential([
        iaa.Sharpen(),
    ])

    # 执行图像增强
    image_augmented_list = []
    for img in image_array:
        img_augmented = aug.augment_images([img])
        image_augmented_list.append(img_augmented[0])

    # 转换为 NumPy 数组
    image_aug = np.array(image_augmented_list)

    # 创建 SimpleITK 图像对象
    sharpened_image = sitk.GetImageFromArray(image_aug)

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 构建输出图像路径
    output_image_path = os.path.join(output_directory, f'Sharpened_5{os.path.basename(image_path)}')

    # 保存增强后的图像
    sitk.WriteImage(sharpened_image, output_image_path)

    print(f"Sharpened image saved at: {output_image_path}")

def Rotate(image_path, output_directory, degree):
    # 读取图像
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # 定义增强器
    aug = iaa.Sequential([
        iaa.Affine(rotate=(degree-1,degree+1), mode='edge', order=0),
    ])

    # 执行图像增强
    image_augmented_list = []
    for img in image_array:
        img_augmented = aug.augment_images([img])
        image_augmented_list.append(img_augmented[0])

    # 转换为 NumPy 数组
    image_aug = np.array(image_augmented_list)

    # 创建 SimpleITK 图像对象
    blurred_image = sitk.GetImageFromArray(image_aug)

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 构建输出图像路径
    output_image_path = os.path.join(output_directory, f'Rotation{degree}_{os.path.basename(image_path)}')

    # 保存增强后的图像
    sitk.WriteImage(blurred_image, output_image_path)

    print(f"Rotated image saved at: {output_image_path}")

degree = -20
image_path = 'D:/Coronary Artery Project/data/adjusted_image/adjusted_cor8.mhd'
image_output_directory = 'D:/Coronary Artery Project/data/image1/H'
mask_path = 'D:/Coronary Artery Project/data/adjusted_mask/corMark8.mhd'
mask_output_directory = 'D:/Coronary Artery Project/data/mask1/H'
Rotate(image_path, image_output_directory, degree)
Rotate(mask_path,mask_output_directory,degree)
# def main():
#
#     image_input_directory = 'D:/Coronary Artery Project/data/adjusted_image'
#     image_output_directory = 'D:/Coronary Artery Project/data/image_aug'
#     # 遍历输入目录中的所有文件
#     for filename in os.listdir(image_input_directory):
#         if filename.lower().endswith('.mhd'):
#             # 构建完整的文件路径
#             image_path = os.path.join(image_input_directory, filename)
#
#             # GaussianBlur_and_save_image(image_path, image_output_directory, 2.5)
#             # AverageBlur_and_save_image(image_path,image_output_directory,9)
#             # MedianBlur_and_save_image(image_path,image_output_directory, 5)
#             MotionBlur_and_save_image(image_path,image_output_directory,k=10,angle=300)
#
#             # Sharpen_and_save_image(image_path,image_output_directory)
#
#             # GaussianNoise_and_save_image(image_path, image_output_directory, 20)
#             # PoissonNoise_and_save_image(image_path, image_output_directory, 8)
#             # SaltAndPepper_and_save_image(image_path, image_output_directory, p=0.05)
#             # LaplaceNoise_and_save_image(image_path,image_output_directory,sd=25)
#             # SaltAndPepper_and_MedianBlur(image_path,image_output_directory,0.3,5)
#             GaussianNoise_and_GaussianBlur(image_path,image_output_directory,20,1.5)
#
#
# if __name__ == '__main__':
#     main()
