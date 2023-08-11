import numpy as np
import os
from PIL import Image
from sklearn.datasets import fetch_openml

if __name__=='__main__':
    digits = fetch_openml(name='mnist_784', version=1)
    datas = digits.data.values.reshape(-1, 28, 28)[2: -2, 2: -2]
    imgs = []
    for data in datas:
        img = Image.fromarray(data)
        img = img.resize((6, 6), Image.BOX)
        imgs.append(np.asarray(img))

    datas = np.asarray(imgs)
    datas = datas.reshape(-1, 36)
    datas = np.delete(np.delete(np.delete(np.delete(datas, 35, 1), 30, 1), 5, 1), 0, 1)
    datas = np.where(datas >= 0.5, 1, 0)
    np.save(os.path.dirname(os.path.abspath(__file__)) + '/processed_mnist.npy', datas)
