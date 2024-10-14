import torch
from torchvision import transforms
import SimpleITK as sitk
import numpy as np
import os

def hflip_and_save(image_path, output_directory):
    # 读取图像
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    transposed_image = image_array.transpose((1, 2, 0))

    # 将int类型的数组转换为float类型
    if transposed_image.dtype == np.uint16:
        transposed_image = transposed_image.astype(np.int16)

    # 定义transforms
    hflip_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=1)
    ])

    # 进行水平翻转
    flipped_tensor_data = hflip_transform(transposed_image)
    flipped_data_array = flipped_tensor_data.numpy()

    # 将NumPy数组转换为SimpleITK图像
    flipped_image = sitk.GetImageFromArray(flipped_data_array)

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 构建输出图像路径
    output_image_path = os.path.join(output_directory, f'hflipped_{os.path.basename(image_path)}')

    # 将翻转后的图像保存到输出目录
    sitk.WriteImage(flipped_image, output_image_path)

def vflip_and_save(image_path, output_directory):
    # 读取图像
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    transposed_image = image_array.transpose((1, 2, 0))

    # 将int类型的数组转换为float类型
    if transposed_image.dtype == np.uint16:
        transposed_image = transposed_image.astype(np.int16)

    # 定义transforms
    vflip_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(p=1)
    ])

    # 进行水平翻转
    flipped_tensor_data = vflip_transform(transposed_image)
    flipped_data_array = flipped_tensor_data.numpy()

    # 将NumPy数组转换为SimpleITK图像
    flipped_image = sitk.GetImageFromArray(flipped_data_array)

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 构建输出图像路径
    output_image_path = os.path.join(output_directory, f'vflipped_{os.path.basename(image_path)}')

    # 将翻转后的图像保存到输出目录
    sitk.WriteImage(flipped_image, output_image_path)

def Rotation_and_save(image_path, output_directory, degree):
    # 读取图像
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    transposed_image = image_array.transpose((1, 2, 0))

    if transposed_image.dtype == np.uint16:
        transposed_image = transposed_image.astype(np.int16)

    # 定义transforms
    Rotation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation((degree-1,degree+1))
        # transforms.Lambda(lambda x: transforms.functional.rotate(x, angle=degree))
    ])

    # 进行水平翻转
    rotated_tensor_data = Rotation_transform(transposed_image)
    rotated_data_array = rotated_tensor_data.numpy()

    # 将NumPy数组转换为SimpleITK图像
    rotated_image = sitk.GetImageFromArray(rotated_data_array)

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 构建输出图像路径
    output_image_path = os.path.join(output_directory, f'Rotation{degree}_{os.path.basename(image_path)}')

    # 将翻转后的图像保存到输出目录
    sitk.WriteImage(rotated_image, output_image_path)

def Erasing_and_save(image_path, output_directory):
    # 读取图像
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    transposed_image = image_array.transpose((1, 2, 0))

    if transposed_image.dtype == np.uint16:
        transposed_image = transposed_image.astype(np.int16)

    # 定义transforms
    Erasing_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(p=1,scale=(0.02,0.1))
    ])

    # 进行水平翻转
    Erased_tensor_data = Erasing_transform(transposed_image)
    Erased_data_array = Erased_tensor_data.numpy()

    # 将NumPy数组转换为SimpleITK图像
    Erased_image = sitk.GetImageFromArray(Erased_data_array)

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 构建输出图像路径
    output_image_path = os.path.join(output_directory, f'Erasing10_{os.path.basename(image_path)}')

    # 将翻转后的图像保存到输出目录
    sitk.WriteImage(Erased_image, output_image_path)


# hflip_and_save(image_path, image_output_directory)
# vflip_and_save(image_path,image_output_directory)
# Rotation_and_save(image_path,image_output_directory,degree)
# Erasing_and_save(image_path,image_output_directory)
degree = 5
mask_path = 'D:/Coronary Artery Project/data/adjusted_mask/corMark2.mhd'
mask_output_directory = 'D:/Coronary Artery Project/data/mask1'
# # hflip_and_save(mask_path, mask_output_directory)
# # vflip_and_save(mask_path,mask_output_directory)
Rotation_and_save(mask_path,mask_output_directory,degree)
# Erasing_and_save(mask_path,mask_output_directory)

# def main():
#
#     degree = -5
#     # image_input_directory = 'D:/Coronary Artery Project/data/adjusted_image'
#     # image_output_directory = 'D:/Coronary Artery Project/data/image1'
#     # # 遍历输入目录中的所有文件
#     # for filename in os.listdir(image_input_directory):
#     #     if filename.lower().endswith('.mhd'):
#     #         # 构建完整的文件路径
#     #         image_path = os.path.join(image_input_directory, filename)
#     #
#     #         Rotation_and_save(image_path, image_output_directory, degree)
#
#     mask_input_directory = 'D:/Coronary Artery Project/data/adjusted_mask'
#     mask_output_directory = 'D:/Coronary Artery Project/data/mask1'
#
#     for filename in os.listdir(mask_input_directory):
#         if filename.lower().endswith('.mhd'):
#             # 构建完整的文件路径
#             mask_path = os.path.join(mask_input_directory, filename)
#
#             Rotation_and_save(mask_path, mask_output_directory, degree)
#
# if __name__ == '__main__':
#     main()
