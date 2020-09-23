import copy
import os
import numpy as np

from keras.models import load_model

MODULE_NAME = "EstimateModel.py"
MODEL_TYPE_LIST = ["value", "updown1", "updown2", "updown3"]
CATEGORICAL_TAG = ["↓", "→", "↑"]
CATEGORICAL_N = 3

class EstimateResultBox():
    def __init__(self, softmaxedData, mostHighData, highProbabilityData):
        self.softmaxedData = softmaxedData
        self.mostHighData = mostHighData
        self.highProbabilityData = highProbabilityData
        


class EstimateModel():
# dataList ... modelName, modelType, modelWindow, normalProb, probList
# probList ... threahold, coverage, probability 順序はthresholdが小さい順になるように
    def __init__(self, dataList):
        self.modelName = dataList[0]
        self.modelType = dataList[1]
        self.modelWindow = dataList[2]
        self.normalProb = dataList[3]
        self.probList = copy.deepcopy(dataList[4])
        
        self.modelPath = __file__.replace(MODULE_NAME, self.modelName)
        
    def initiate(self):
        #modelの初期化。estimateの等の前にこれが必要
        self.model = load_model(self.modelPath)
        return self.model
    
    def estimate(self, data):
        #dataを入力すると、予測をprintし、結果をEstimateResultBoxで返す。
        #dataは１単位分を想定
        data = data.reshape((1, ) + data.shape)
        data = self.model.predict(data)
        result = None
        if(self.modelType == MODEL_TYPE_LIST[0]):
            pass
            
        else:
            result = self.processData(data.reshape(-1))
        
        self.printResult(result)
        
        return result           
        
        
    def processData(self, data):
        #predictから返されたデータをもとに整理したデータをEstimateResultBoxで返す
        #
        #softmaxedData...すべてをsoftmax化したもの
        #mostHighData...一番高いところのsoftmax値とインデックス
        #highProbabilityData...確度が高いかのデータ checkProbability参照.
        
        softmaxedData = self.softmaxAll(data)
        mostHighData = np.array([self.softmaxOnly(data), np.argmax(i)])
        highProbabilityData = self.checkProbability(mostHighData[0])
        
        res = EstimateResultBox(softmaxedData, mostHighData, highProbabilityData)
        return res
        
    def checkProbability(mostHighProbability):
        #確度が高いかを返す
        #特にないならNone, 高いと、
        #該当するprobListのインデックス、coverage, probabilityのリストを返す。
        result = None
        if(self.probList):
            i = 0
            for ls in self.probList:
                threashold = ls[0]
                coverage = ls[1]
                probability = ls[2]
                if(mostHighProbability > threashold):
                    result = [i, coverage, probability]
                    i += 1
                else:
                    break
        return result
    
    def printResult(self, result):
        #EstimateResultBoxを受け取り、printする
        print("{}: +{}".format(self.modelType, self.modelWindow))
        for i in range(CATEGORICAL_N):
            print("{}: {:.0%} , ".format(CATEGORICAL_TAG[i], result.softmaxedData[i]), end = "")
        print()
        
        additional = highProbabilityFormat(result.highProbabilityData)
        value = result.mostHighData[0]
        arrow = CATEGORICAL_TAG[result.mostHighData[1]]
        
        print("◆Ans: {} {:.0%}  {}".format(arrow, value, additional))
        
    
    def highProbabilityFormat(highProbabilityData):
        #今回のデータが確度が高いときに表示する文字列を作る
        result = " - "
        if(highProbabilityData):
            indexN = highProbabilityData[0]
            probability = highProbability[2]
            
            result = "★" * (indexN + 1)
            result += " ({:.1%})".format(probability)
        
        return result
            
           

    def softmaxOnly(lis):
        #一番大きなインデックスのsoftmax値のみを返す
        denom = sum([math.exp(i) for i in lis])
        return math.exp(max(lis)) / denom

    def softmaxAll(lis):
        #すべてのsoftmax値を返す
        denom = sum([math.exp(i) for i in lis])
        return np.array([math.exp(i) / denom for i in lis])
    

        
        