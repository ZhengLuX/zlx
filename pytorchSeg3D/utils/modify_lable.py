import os
import SimpleITK as sitk

def merge_categories(mask_path, output_path, categories_to_merge, new_category):
    # 读取图像
    mask = sitk.ReadImage(mask_path)

    mask_array = sitk.GetArrayFromImage(mask)

    # 遍历数组并重新标记类别
    for i in range(mask_array.shape[0]):
        for j in range(mask_array.shape[1]):
            for k in range(mask_array.shape[2]):
                if mask_array[i, j, k] in categories_to_merge:
                    mask_array[i, j, k] = new_category

    # 将修改后的数组转换回SimpleITK图像
    new_mask = sitk.GetImageFromArray(mask_array)
    new_mask.CopyInformation(mask)

    # 保存新的图像
    sitk.WriteImage(new_mask, output_path)

def process_directory(root_dir, output_base_dir, categories_to_merge, new_category):
    for subdir, dirs, files in os.walk(root_dir):
        # 获取当前子文件夹的名称
        subdir_name = os.path.basename(subdir)
        # 在输出基础目录下创建与当前子文件夹同名的文件夹
        output_subdir = os.path.join(output_base_dir, subdir_name)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        for file in files:
            if file.endswith('.mhd'):
                mask_path = os.path.join(subdir, file)
                output_path = os.path.join(output_subdir, file)  # 使用原始文件名
                merge_categories(mask_path, output_path, categories_to_merge, new_category)

# 定义根目录和输出基础目录
root_dir = '/home/zlx/PycharmProjects/pytorch_segment3D/data/mask_aug'  # 替换为你的根目录路径
output_base_dir = '/home/zlx/PycharmProjects/pytorch_segment3D/data/mm'  # 替换为你的输出基础目录路径

# 定义要合并的类别和新类别
categories_to_merge = [1, 2, 3]  # 替换为你要合并的类别
new_category = 1  # 替换为合并后的新类别

# 处理目录
process_directory(root_dir, output_base_dir, categories_to_merge, new_category)

