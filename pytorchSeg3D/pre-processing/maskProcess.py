import os
import SimpleITK as sitk
from utils.modify_mask import modify_categories

def process_batch(root_dir, output_base_dir, categories_to_modify, new_category):
    for subdir, dirs, files in os.walk(root_dir):
        # 获取当前子文件夹的名称
        subdir_name = os.path.basename(subdir)
        # 在输出基础目录下创建与当前子文件夹同名的文件夹
        output_subdir = os.path.join(output_base_dir, subdir_name)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        for file in files:
            if file.endswith(('.mhd','.label.nii.gz')):
                mask_path = os.path.join(subdir, file)
                output_path = os.path.join(output_subdir, file)  # 使用原始文件名

                # 读取mhd文件
                itk_image = sitk.ReadImage(mask_path)
                image_array = sitk.GetArrayFromImage(itk_image)

                # 去除不需要的类别
                processed_array = modify_categories(image_array, categories_to_modify, new_category)

                # 将numpy数组转换回SimpleITK图像并保存
                itk_image = sitk.GetImageFromArray(processed_array)
                itk_image.CopyInformation(itk_image)  # 保留原始图像的元数据
                sitk.WriteImage(itk_image, output_path)

# 定义根目录和输出基础目录
# root_dir = 'D:/Coronary Artery Project/data/mask'  # 替换为你的根目录路径
root_dir ='D:/ImageCAS/10'
output_base_dir = 'D:/ImageCAS/label'  # 替换为你的输出基础目录路径

categories_to_modify = [1]  # 要去除的类别列表

new_category = [2]

process_batch(root_dir, output_base_dir, categories_to_modify, new_category)



