import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
def high_rel_res(test_inputs, test_outputs, model, threshold):
    res = high_rel_pred(test_inputs, model, threshold)
    coverage, acc = high_rel_acc(res, test_outputs)
    if(acc == -1):
        print("No data has high reliability.")
    print("coverage : " + str(coverage) + ", accuracy : " + str(acc))

def high_rel_acc(res, test_outputs):
    length = res.shape[0]
    num = np.count_nonzero(res >= 0)
    lis = [res[i] for i in range(length) if res[i] >= 0 and res[i] == test_outputs[i]]
    lis_n = len(lis)
    if num == 0:
        return [0, -1]
    else:
        return [num / length, lis_n / num]

def high_rel_pred(test_inputs, model, threshold):
    lis = model.predict(test_inputs)
    probs = prob_model(lis)
    length = probs[0].shape[0]
    result = np.array([probs[1][i] if probs[0][i] > threshold else -1\
                       for i in range(length) ])
    return result

def probs_model(result):
    lis1 = np.array([softmax_all(i) for i in result])
    lis2 = np.array([np.argmax(i) for i in result])
    return [lis1, lis2]


def prob_model(result):
    lis1 = np.array([softmax_only(i) for i in result])
    lis2 = np.array([np.argmax(i) for i in result])
    return [lis1, lis2]

def softmax_only(lis):
    denom = sum([math.exp(i) for i in lis])
    return math.exp(max(lis)) / denom
    
def softmax_all(lis):
    denom = sum([math.exp(i) for i in lis])
    return np.array([math.exp(i) / denom for i in lis])

def heatmaps(data, imageName, isShown = True):
    #dataはいまのところ四次元np.arrayで
    #labelsは、dataのparamをそれぞれ上の階層の値からリストにしたもの
    shape = data.shape
    fig = plt.figure(figsize = (shape[0]*5, shape[1]*1))
    
    limits = [np.min(data), np.max(data)]
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            innerData = data[i, j, :, :]
            ax = fig.add_subplot(shape[0], shape[1], i*shape[1] + j+1)
            im = ax.imshow(innerData, interpolation = "nearest", cmap = "jet", aspect=0.5, alpha = 0.5,\
                           norm=Normalize(vmin=limits[0], vmax=limits[1]))
            ys, xs = np.meshgrid(range(innerData.shape[0]), range(innerData.shape[1]), indexing = "ij")
            for (x, y, val) in zip(xs.flatten(), ys.flatten(), innerData.flatten()):
                ax.text(x, y, "{0:.4f}".format(val), horizontalalignment="center",\
                         verticalalignment="center")
    plt.savefig(imageName)
    if(isShown):
        plt.show()

    return