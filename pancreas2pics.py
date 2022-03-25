import numpy as np
from PIL import Image
import SimpleITK as sitk
import os


def getLabelPositionShape(CTimage_path):
    image = sitk.ReadImage(CTimage_path)
    array = sitk.GetArrayFromImage(image)
    (x, y, z) = array.shape
    position = [0, 1, 2, 3, 4, 5]
    height = x
    # print(x)
    shape = [0, 1, 2]

    first = False
    last = False
    for i in range(x):
        if array[i, :, :].max() != 0 and first == False:
            first = True
            # position.append(i)
            position[0] = i
        if array[i, :, :].max() == 0 and first == True and last == False:
            last = True
            # position.append(i)
            position[1] = i
            break
    shape[0] = position[1] - position[0]

    first = False
    last = False
    for i in range(y):
        if array[:, i, :].max() != 0 and first == False:
            first = True
            position[2] = i
        if array[:, i, :].max() == 0 and first == True and last == False:
            last = True
            position[3] = i
            break
    shape[1] = position[3] - position[2]

    first = False
    last = False
    for i in range(z):
        if array[:, :, i].max() != 0 and first == False:
            first = True
            position[4] = i
        if array[:, :, i].max() == 0 and first == True and last == False:
            last = True
            position[5] = i
            break
    shape[2] = position[5] - position[4]

    return position, shape, height


def toBmp(imgfile, labelfile, direc):
    title = 0
    num_dicts = {'IPMN': [3, 3, 2], "NET": [4, 4, 2], "PDAC": [1, 2, 2], "SCN": [5, 4, 2]}
    pic_num = [3, 3, 3]
    p, s, height = getLabelPositionShape(labelfile)
    image = sitk.ReadImage(imgfile)
    array = sitk.GetArrayFromImage(image)
    height = array.shape[0]
    ratio = array.shape[1]
    zeros = np.zeros((ratio - height, ratio, ratio))
    zeros = np.concatenate((array, zeros), axis=0)
    # 定义三个维度的随机点的列表
    ran_x = []
    ran_y = []
    ran_z = []
    # 三个维度的点的进入列表
    for k in range(pic_num[0]):
        ran_x.append(p[0] + s[0] * k / pic_num[0])
    for k in range(pic_num[1]):
        ran_y.append(p[2] + s[1] * k / pic_num[1])
    for k in range(pic_num[2]):
        ran_z.append(p[4] + s[2] * k / pic_num[2])

    for m in ran_x:
        for n in ran_y:
            for o in ran_z:
                arrayx = zeros[int(m)]
                arrayy = zeros[:, int(n)]
                arrayz = zeros[:, :, int(o)]
                b = np.concatenate((arrayx.reshape(ratio, ratio, 1), arrayy.reshape(ratio, ratio, 1), arrayz.reshape(ratio, ratio, 1)), axis=2)
                im = Image.fromarray(np.uint8(b))
                im.save(os.path.join(direc, f"c{title}.bmp"))
                title = title + 1


def inp(nam):  # A B C.nii.gz
    try:
        na = nam
        nam = nam.replace("nii.gz", "").replace(".", "")
        picpath = f'intermediate/pics/{nam}'
        if not os.path.exists(picpath):
            os.mkdir(picpath)
        toBmp(os.path.join('intermediate/image/', na), os.path.join('intermediate/label/', na), picpath)
        return {"status": True}
    except Exception as e:
        return {"status": False, "message": e}


if __name__ == "__main__":
    print("pancreas->pics")
    rootDir = 'intermediate/image/'
    for item in os.listdir(rootDir):
        print(item)
        inp(item)
