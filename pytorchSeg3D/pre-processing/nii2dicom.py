import time,os
from utils.formatConvert import convert as img_convert
from utils.mask2dicom import convert as msk_convert
from utils.mask2dicom import convert_list as msk_list_convert

def nifti2dcm(input_nifti_dir, output_dicom_dir, struct_name, roi_list):
    head, tail = os.path.split(input_nifti_dir)
    patienName = tail
    patienID = time.strftime("%Y%m%d%H%M%S")+'_'+patienName
    img_path = os.path.join(input_nifti_dir,'11.img.nii.gz')
    msk_path = os.path.join(input_nifti_dir,'11.label.nii.gz')
    img_convert(patienID, patienName, img_path, output_dicom_dir)
    print('Image conversion completed. ')
    msk_convert(msk_path, output_dicom_dir, output_dicom_dir, struct_name, roi_list)
    print('Label conversion completed. ')

if __name__ == "__main__":

    input_nifti_dir = r'D:/ImageCAS/1-200'
    output_dicom_dir = r'D:/ImageCAS/out'
    struct_name = 'datu'
    roi_list=None
    nifti2dcm(input_nifti_dir, output_dicom_dir, struct_name, roi_list)
