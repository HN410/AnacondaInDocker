import numpy as np
import math
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

