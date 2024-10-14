import SimpleITK as sitk
import os
import numpy as np
def extract_roi(image_path, label_path, output_folder):
    """
    从mha文件中提取ROI区域，并保存到指定文件夹中。

    参数:
    image_path: 原始图像的路径。
    label_path: 标签图像的路径。
    output_folder: 输出ROI图像的文件夹路径。
    """
    # 读取图像和标签
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    # 将标签图像中的非零像素设置为1
    label_binary = sitk.BinaryThreshold(label, lowerThreshold=0, upperThreshold=float('inf'), insideValue=1,
                                        outsideValue=0)

    # 确保标签掩码与原始图像具有相同的空间信息
    label_binary.CopyInformation(image)

    # 将标签图像转换为相同类型和尺寸的掩码
    label_mask = sitk.Cast(label_binary, image.GetPixelID())

    # 提取ROI
    roi = image * label_mask

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取原始图像的文件名
    image_filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, image_filename)

    # 保存ROI
    sitk.WriteImage(roi, output_path)


# 使用函数示例
image_path = 'D:/Coronary Artery Project/data2/image_aug/A/A_1.mhd'
label_path = 'D:/Coronary Artery Project/data2/mask_aug/A/A_1.mhd'
roi_label_values = [1, 2, 3]  # 要提取的标签值列表
output_folder = 'D:/Coronary Artery Project/data2/ROI/image/'  # 提取的ROI保存文件夹路径

extract_roi(image_path, label_path, output_folder)