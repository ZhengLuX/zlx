import os
import SimpleITK as sitk
import logging
from utils.dataProcess import resize_volume,adjust_contrast,adjust_spacing
# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 函数映射，允许用户通过名称选择要应用的函数
image_function_map = {
    'resize_image': resize_volume,
    'adjust_spacing': adjust_spacing,
    'adjust_image_contrast': adjust_contrast
}

mask_function_map = {
    'resize_mask': resize_volume,
    'adjust_spacing': adjust_spacing
}

def apply_functions(image, function_map, functions_to_apply, **kwargs):
    for function_name in functions_to_apply:
        if function_name in function_map:
            image = function_map[function_name](image, **kwargs)
        else:
            logging.error(f"Function '{function_name}' is not supported.")
            return None
    return image


def process_image_and_mask(image_dir, mask_dir, output_image_dir, output_mask_dir, image_processing_steps, mask_processing_steps, **kwargs):
    """
    处理图像和掩码
    """
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.mhd'):
            image_path = os.path.join(image_dir, image_file)
            mask_file = image_file  # 假设图像文件和掩码文件有相同的文件名
            mask_path = os.path.join(mask_dir, mask_file)

            try:
                # 读取图像和掩码
                itk_image = sitk.ReadImage(image_path)
                itk_mask = sitk.ReadImage(mask_path)

                # 应用图像处理步骤
                for step in image_processing_steps:
                    step_name = step['name']
                    step_params = step.get('params', {})
                    itk_image = apply_functions(itk_image, image_function_map, [step_name], **step_params)

                # 应用掩码处理步骤
                for step in mask_processing_steps:
                    step_name = step['name']
                    step_params = step.get('params', {})
                    itk_mask = apply_functions(itk_mask, mask_function_map, [step_name], **step_params)

                # 保存处理后的图像和掩码
                output_image_path = os.path.join(output_image_dir, image_file)
                output_mask_path = os.path.join(output_mask_dir, mask_file)

                sitk.WriteImage(itk_image, output_image_path)
                sitk.WriteImage(itk_mask, output_mask_path)

                logging.info(f"Processed image and mask saved to {output_image_path} and {output_mask_path}")
            except Exception as e:
                logging.error(f"Error processing {image_path} or {mask_path}: {e}")

def main(root_dir, output_dir):
    # 创建输出目录
    output_image_dir = os.path.join(output_dir, 'image')
    output_mask_dir = os.path.join(output_dir, 'mask')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # 定义图像和掩码的处理步骤
    image_processing_steps = [
        {'name': 'resize_image', 'params': {'new_shape': (128, 128, 128)}},
        {'name': 'adjust_spacing', 'params': {'new_spacing': (1.0, 1.0, 1.0)}},
        # {'name': 'adjust_image_contrast', 'params': {'window_level': 170, 'window_width': 600}}
    ]

    mask_processing_steps = [
        {'name': 'resize_mask', 'params': {'new_shape': (128, 128, 128)}},
        {'name': 'adjust_spacing', 'params': {'new_spacing': (1.0, 1.0, 1.0)}}
    ]

    # 定义输入目录
    image_dir = os.path.join(root_dir, 'image')
    mask_dir = os.path.join(root_dir, 'mask')

    # 处理图像和掩码
    process_image_and_mask(image_dir, mask_dir, output_image_dir, output_mask_dir, image_processing_steps, mask_processing_steps)

if __name__ == "__main__":
    root_dir = '/home/zlx/PycharmProjects/pytorch_segment3D/data/Aorta/256z/512xy/originalData/Crop/256xyz'
    output_dir = '/home/zlx/PycharmProjects/pytorch_segment3D/data/Aorta/256z/512xy/originalData/Crop/256xyz/Resized128xyz'
    main(root_dir, output_dir)