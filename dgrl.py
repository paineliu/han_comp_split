# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
import struct
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm


def read_from_dgrl(dgrl):
    if not os.path.exists(dgrl):
        print('DGRL not exis!')
        return

    dir_name, base_name = os.path.split(dgrl)
    label_dir = dir_name+'_label'
    image_dir = dir_name+'_images'
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    with open(dgrl, 'rb') as f:
        # 读取表头尺寸
        header_size = np.fromfile(f, dtype='uint8', count=4)
        header_size = sum([j << (i*8) for i, j in enumerate(header_size)])
        # print(header_size)

        # 读取表头剩下内容，提取 code_length
        header = np.fromfile(f, dtype='uint8', count=header_size-4)
        code_length = sum([j << (i*8) for i, j in enumerate(header[-4:-2])])
        # print(code_length)

        # 读取图像尺寸信息，提取图像中行数量
        image_record = np.fromfile(f, dtype='uint8', count=12)
        height = sum([j << (i*8) for i, j in enumerate(image_record[:4])])
        width = sum([j << (i*8) for i, j in enumerate(image_record[4:8])])
        line_num = sum([j << (i*8) for i, j in enumerate(image_record[8:])])
        print('图像尺寸:')
        print(height, width, line_num)

        # 读取每一行的信息
        for k in range(line_num):
            print(k+1)

            # 读取该行的字符数量
            char_num = np.fromfile(f, dtype='uint8', count=4)
            char_num = sum([j << (i*8) for i, j in enumerate(char_num)])
            print('字符数量:', char_num)

            # 读取该行的标注信息
            label = np.fromfile(f, dtype='uint8', count=code_length*char_num)
            label = [label[i] << (8*(i % code_length))
                     for i in range(code_length*char_num)]
            label = [sum(label[i*code_length:(i+1)*code_length])
                     for i in range(char_num)]
            label = [struct.pack('I', i).decode(
                'gbk', 'ignore')[0] for i in label]
            print('合并前：', label)
            label = ''.join(label)
            # 去掉不可见字符 \x00，这一步不加的话后面保存的内容会出现看不见的问题
            label = ''.join(label.split(b'\x00'.decode()))
            print('合并后：', label)

            # 读取该行的位置和尺寸
            pos_size = np.fromfile(f, dtype='uint8', count=16)
            y = sum([j << (i*8) for i, j in enumerate(pos_size[:4])])
            x = sum([j << (i*8) for i, j in enumerate(pos_size[4:8])])
            h = sum([j << (i*8) for i, j in enumerate(pos_size[8:12])])
            w = sum([j << (i*8) for i, j in enumerate(pos_size[12:])])
            # print(x, y, w, h)

            # 读取该行的图片
            bitmap = np.fromfile(f, dtype='uint8', count=h*w)
            bitmap = np.array(bitmap).reshape(h, w)

            # 保存信息
            label_file = os.path.join(
                label_dir, base_name.replace('.dgrl', '_'+str(k)+'.txt'))
            with open(label_file, 'w') as f1:
                f1.write(label)
            bitmap_file = os.path.join(
                image_dir, base_name.replace('.dgrl', '_'+str(k)+'.jpg'))
            cv.imwrite(bitmap_file, bitmap)


if __name__ == '__main__':
    dgrl_paths = Path('dataset/HandWriteDBRecog/HWDB2.2Train').iterdir()
    dgrl_paths = list(dgrl_paths)
    for dgrl_path in tqdm(dgrl_paths):
        read_from_dgrl(dgrl_path)

