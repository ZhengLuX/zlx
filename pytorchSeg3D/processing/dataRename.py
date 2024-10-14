import os

def batch_rename_subfolder_files(root_directory):
    # 遍历根目录下的所有子文件夹
    for subfolder in os.listdir(root_directory):
        subfolder_path = os.path.join(root_directory, subfolder)
        if os.path.isdir(subfolder_path):
            # 获取子文件夹中的所有.mhd文件
            mhd_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.mhd')]
            # 从1开始对文件进行编号
            for index, mhd_filename in enumerate(mhd_files, start=41):
                # 构建新的文件名
                prefix = subfolder + '_'
                new_mhd_filename = f"{prefix}{index}.mhd"
                raw_filename = mhd_filename[:-4] + '.raw'
                new_raw_filename = f"{prefix}{index}.raw"

                # 构建完整的文件路径
                old_mhd_file_path = os.path.join(subfolder_path, mhd_filename)
                new_mhd_file_path = os.path.join(subfolder_path, new_mhd_filename)
                old_raw_file_path = os.path.join(subfolder_path, raw_filename)
                new_raw_file_path = os.path.join(subfolder_path, new_raw_filename)

                # 重命名文件
                os.rename(old_mhd_file_path, new_mhd_file_path)
                os.rename(old_raw_file_path, new_raw_file_path)

                # 更新.mhd文件内部指向.raw文件的路径
                update_mhd_file(new_mhd_file_path, new_raw_filename)

                print(f"Renamed '{mhd_filename}' to '{new_mhd_filename}' and '{raw_filename}' to '{new_raw_filename}' in '{subfolder}'")

def update_mhd_file(mhd_file_path, new_raw_filename):
    # 读取.mhd文件内容
    with open(mhd_file_path, 'r') as file:
        lines = file.readlines()

    # 更新指向.raw文件的路径
    with open(mhd_file_path, 'w') as file:
        for line in lines:
            if line.lower().startswith('elementdatafile ='):
                line = f"ElementDataFile = {new_raw_filename}\n"
            file.write(line)

image_root_directory = 'D:/Coronary Artery Project/data/image1'  # 替换为您的根目录路径
mask_root_directory = 'D:/Coronary Artery Project/data/mask1'  # 替换为您的根目录路径
batch_rename_subfolder_files(image_root_directory)
batch_rename_subfolder_files(mask_root_directory)