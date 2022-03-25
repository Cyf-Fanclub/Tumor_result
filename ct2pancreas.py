import numpy as np
import SimpleITK as sitk
import os

# 切割大小，半径
cut_size = 100


def cutting(item, image_path, label_path, outp_image_path, outp_label_path):
    try:
        image = sitk.ReadImage(label_path)
    except:
        return {"status": False, "message": f"无法打开label:{label_path}"}
    array = sitk.GetArrayFromImage(image)
    # 计算平均位置
    frame_num, width, height = array.shape
    height_avg = 0
    num_avg = 0
    width_avg = 0
    # print(array.shape)

    arr1, arr2, arr3 = np.where(array != 0)
    for i in range(len(arr1)):
        num_avg += arr1[i]
        width_avg += arr2[i]
        height_avg += arr3[i]
    tot_num = len(arr1)
    if tot_num == 0:
        return {"status": False, "message": f"label处理错误:arr长度为0"}
    num_avg /= tot_num
    num_avg = round(num_avg)
    width_avg /= tot_num
    width_avg = round(width_avg)
    height_avg /= tot_num
    height_avg = round(height_avg)
    # print(num_avg,width_avg,height_avg)

    num_left = 0
    if num_avg - cut_size < 0:
        num_left = cut_size - num_avg
    num_right = 2 * cut_size
    if cut_size + num_avg >= frame_num:
        num_right = frame_num - num_avg + cut_size

    width_left = 0
    if width_avg - cut_size < 0:
        width_left = cut_size - width_avg
    width_right = 2 * cut_size
    if cut_size + width_avg >= width:
        width_right = width - width_avg + cut_size

    height_left = 0
    if height_avg - cut_size < 0:
        height_left = cut_size - height_avg
    height_right = 2 * cut_size
    if cut_size + height_avg >= height:
        height_right = height - height_avg + cut_size
    num_left = int(num_left)
    num_right = int(num_right)

    width_left = int(width_left)
    width_right = int(width_right)

    height_left = int(height_left)
    height_right = int(height_right)

    l1 = int(max(0, num_avg - cut_size))
    r1 = int(min(frame_num, cut_size + num_avg))
    l2 = int(max(0, width_avg - cut_size))
    r2 = int(min(width, cut_size + width_avg))
    l3 = int(max(0, height_avg - cut_size))
    r3 = int(min(height, cut_size + height_avg))
    label_array = np.zeros((2 * cut_size, 2 * cut_size, 2 * cut_size))
    label_array[num_left:num_right, width_left:width_right, height_left:height_right] = array[l1:r1, l2:r2, l3:r3]

    try:
        image = sitk.ReadImage(image_path)
    except:
        return {"status": False, "message": f"无法打开image:{image_path}"}
    array = sitk.GetArrayFromImage(image)

    out_image = sitk.GetImageFromArray(label_array)
    sitk.WriteImage(out_image, os.path.join(outp_label_path, item))

    image_array = np.zeros((2 * cut_size, 2 * cut_size, 2 * cut_size))
    image_array[num_left:num_right, width_left:width_right, height_left:height_right] = array[l1:r1, l2:r2, l3:r3]

    out_image = sitk.GetImageFromArray(image_array)
    sitk.WriteImage(out_image, os.path.join(outp_image_path, item))
    return {"status": True}


inp_image_path = r'input/image/'
inp_label_path = r'input/label/'

outp_image_path = r'intermediate/image/'
outp_label_path = r'intermediate/label/'


def inp(item):  # A B C.nii.gz
    image_path = os.path.join(inp_image_path, item)
    label_path = os.path.join(inp_label_path, item)
    if os.path.exists(label_path):
        return cutting(item, image_path, label_path, outp_image_path, outp_label_path)
    else:
        return {"status": False, "message": f"不存在label:{label_path}"}


if __name__ == "__main__":
    print("ct->pancreas")
    for item in os.listdir(inp_image_path):
        print(item)
        inp(item)
