import ct2pancreas
import pancreas2pics
import pics2result
import os
import shutil
import pandas as pd
#import openpyxl
import time


def makeFolder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def makeFolders():
    makeFolder('input/')
    makeFolder('input/image/')
    makeFolder('input/label/')

    makeFolder('intermediate/')
    makeFolder('intermediate/image/')
    makeFolder('intermediate/label/')
    makeFolder('intermediate/pics/')

    makeFolder('output/')


def process(item):
    nam = item.split('.')[0]
    print(f"\n{nam}")

    print('ct->pancreas')
    result = ct2pancreas.inp(item)
    print(result)
    if result["status"] == False:
        return result

    print('pancreas->pics')
    result = pancreas2pics.inp(item)
    print(result)
    if result["status"] == False:
        return result

    print('pics->result')
    result = pics2result.inp(nam)
    print(result)

    return result


if __name__ == "__main__":
    makeFolders()
    while True:
        switch = input("\n\n使用单次运行模式请输入1\n使用批处理模式请输入2\n退出请输入3\n")
        if switch == "3":
            break
        elif switch == "2":
            inp_image_path = r'input/image/'
            data = []
            for item in os.listdir(inp_image_path):
                result = process(item)
                if result["status"] == False:
                    data.append([item, "", "", "", "", "Error"])
                else:
                    r = result["message"]
                    r_formatted = [item, r["IPMN"], r["NET"], r["PDAC"], r["SCN"], max(r, key=r.get)]
                    #print(r_formatted)
                    data.append(r_formatted)
            #print(data)
            df = pd.DataFrame(data, columns=['NAME', 'IPMN', 'NET', 'PDAC', 'SCN', 'Result'])
            df.to_excel(f'output/result{int(time.time())}.xlsx', index=False)

        elif switch == "1":
            inp_image_path = input("请输入病人完整CT图路径(.nii.gz)：")
            if not os.path.exists(inp_image_path):
                print(f'File does not exist：{inp_image_path}')
                continue
            item = inp_image_path.split('/')[-1].split('\\')[-1]  # A B C.nii.gz
            try:
                shutil.copyfile(inp_image_path, os.path.join('input/image/', item))
            except Exception as e:
                pass

            inp_label_path = input("请输入病人肿瘤标记路径(.nii.gz)（若无请按回车）：")
            if inp_label_path.strip() == "":
                inp_label_path = os.path.join('input/label/', item)
            if not os.path.exists(inp_label_path):  # 只存在CT不存在label
                print(f'File does not exist：{inp_label_path}')  # 改为调用自动裁剪肿瘤模块
                continue
            try:
                shutil.copyfile(inp_label_path, os.path.join('input/label/', item))
            except Exception as e:
                pass
            process(item)