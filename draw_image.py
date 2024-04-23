import os
import numpy as np
import h5py
import json
import time
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt


# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r", encoding='utf-8') as f:
        jdata = json.load(f)
        train_acc = []
        train_loss = []
        test_loss = []
        for item in jdata['epoch']:
            acc = item['test_acc']
            tlost = item['train_loss']
            vlost = item['valid_loss']
            train_acc.append(acc)
            train_loss.append(tlost)
            test_loss.append(vlost)

        

    return np.asfarray(train_acc, float), np.asfarray(train_loss, float), np.asfarray(test_loss, float)


# 不同长度数据，统一为一个标准，倍乘x轴
def multiple_equal(x, y):
    x_len = len(x)
    y_len = len(y)
    times = x_len/y_len
    y_times = [i * times for i in y]
    return y_times

def draw_acc(json_filename, img_filename):
    train_acc_path = json_filename

    # y_train_loss = data_read(train_loss_path)
    y_train_acc, y_train_loss , y_test_loss = data_read(train_acc_path)

    x_train_loss = range(len(y_train_loss))
    x_train_acc = multiple_equal(x_train_loss, range(len(y_train_acc)))

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')
    plt.ylabel('accuracy')
    plt.ylim((0, 1))
    # plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.plot(x_train_acc, y_train_acc, linestyle="solid", label="test accuracy")
    plt.legend()

    plt.title('Accuracy curve')
    # plt.show()
    plt.savefig(img_filename)
    plt.close()


def draw_loss(json_filename, img_filename):
    train_acc_path = json_filename

    # y_train_loss = data_read(train_loss_path)
    y_train_acc, y_train_loss , y_test_loss = data_read(train_acc_path)

    x_train_loss = range(len(y_train_loss))
    x_train_acc = multiple_equal(x_train_loss, range(len(y_train_acc)))

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')
    plt.ylabel('loss')

    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.plot(x_train_acc, y_test_loss,  color='red', linestyle="solid", label="test loss")

    plt.legend()

    plt.title('Loss curve')
    # plt.show()
    plt.savefig(img_filename)
    plt.close()

if __name__ == "__main__":

    draw_acc('./output/han_comp_casia/han_comp_casia_model.pt.json', 'img_comp_acc.png')
    draw_acc('./output/han_sorder_palm_4f60/han_sorder_palm_4f60_model.pt.json', 'img_sorder_4f60_acc.png')
    draw_loss('./output/han_comp_casia/han_comp_casia_model.pt.json', 'img_comp_loss.png')
    draw_loss('./output/han_sorder_palm_4f60/han_sorder_palm_4f60_model.pt.json', 'img_sorder_4f60_loss.png')
