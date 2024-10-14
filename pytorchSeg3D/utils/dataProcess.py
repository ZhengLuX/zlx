import SimpleITK as sitk
import os
import numpy as np

def read_mhd_and_raw(file_path):
    itk_image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(itk_image)
    return image_array

def resize_volume(input_image, new_shape=(512, 512, 256), resample_method=sitk.sitkNearestNeighbor):
    """
    使用SimpleITK的ResampleImageFilter调整图像尺寸，并返回调整后的图像。

    参数:
    input_image: 输入的SimpleITK图像对象。
    new_shape: 输出图像的新尺寸，格式为(height, width, depth)。
    resample_method: 采样方法，默认为sitk.sitkNearestNeighbor。
    """
    # 创建ResampleImageFilter对象
    resampler = sitk.ResampleImageFilter()
    origin_size = input_image.GetSize()  # 原来的体素块尺寸
    origin_spacing = input_image.GetSpacing()
    new_size = np.array(new_shape, dtype=float)
    factor = origin_size / new_size
    new_spacing = origin_spacing * factor
    new_size = new_size.astype(np.int32)  # spacing不能是整数，但size必须是整数

    # 设置重采样参数
    resampler.SetReferenceImage(input_image)  # 设置参考图像
    resampler.SetSize(new_size.tolist())  # 设置输出尺寸
    resampler.SetOutputSpacing(new_spacing.tolist())  # 设置输出间距
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))  # 设置变换
    resampler.SetInterpolator(resample_method)  # 设置插值方法

    # 执行重采样
    itk_image_resampled = resampler.Execute(input_image)

    # 返回处理后的图像
    return itk_image_resampled


def adjust_spacing(input_image, new_spacing):
    """
    调整图像的spacing而不改变图像内容。

    参数:
    input_image: 输入的SimpleITK图像对象。
    new_spacing: 新的spacing，格式为(spacing_x, spacing_y, spacing_z)。

    返回:
    调整了spacing的SimpleITK图像对象。
    """
    if not isinstance(new_spacing, tuple) or len(new_spacing) != input_image.GetDimension():
        raise ValueError("new_spacing must be a tuple with the same number of elements as the image dimension")

    # 创建一个与原始图像相同内容的新图像
    adjusted_image = sitk.Image(input_image)

    # 设置新图像的spacing
    adjusted_image.SetSpacing(new_spacing)

    return adjusted_image

def adjust_window(image, window_level=170, window_width=600):
    """
        调整图像的对比度

        参数:
        image：要调整对比度的原始图像
        window_level: 窗位，默认为170。
        window_width: 窗宽，默认为600。
        """

    # 计算窗位对应的实际最小值和最大值
    window_min = window_level - window_width / 2.0
    window_max = window_level + window_width / 2.0

    # 创建窗位和窗宽滤波器
    window_image_filter = sitk.IntensityWindowingImageFilter()
    window_image_filter.SetWindowMinimum(window_min)
    window_image_filter.SetWindowMaximum(window_max)
    window_image_filter.SetOutputMinimum(0)
    window_image_filter.SetOutputMaximum(255)

    # 应用窗位和窗宽滤波器
    windowed_image = window_image_filter.Execute(image)

    return windowed_image