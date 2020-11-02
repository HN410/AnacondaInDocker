import numpy as np
import datetime
import pandas as pd
import random
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc

COLUMNS_NAME = ["index", "通貨ペア", "日付", "始値(Bid)", "高値(Bid)", "安値(Bid)" ,"終値(Bid)"]
currencyName = "USD/JPY"
startTime = datetime.datetime.now()

def easyDataMaker(winLen, changePointList, noise = 0, csvName = "", visualize = False):
    #カリキュラムラーニング用の簡単なデータを作る
    #winLen データ長
    #changePointList 増減が転換する時間
    #hasNoise ノイズの大きさ
    head = getHeadData(winLen)
    main = MainDataMaker().getMain(winLen, changePointList, noise)
    res = pd.concat([head, main], axis = 1)
    
    if(csvName):
        res.drop("index", axis = 1).to_csv(csvName, encoding='utf-8')
        l = None
        with open(csvName, encoding = "utf-8") as f:
                    l = f.readlines()
        l[0] = "index" + l[0]
        with open(csvName, "w", encoding = "utf-8") as f:
            f.writelines(l)

    if(visualize):
        candleVisualize(res)
        
    return res
    
def getHeadData(winLen):
    head = [[str(i), currencyName,
             getShiftedTimeStr(startTime, i)]
            for i in range(winLen)]
    res = pd.DataFrame(head, columns = COLUMNS_NAME[:3])
    return res
    
def getShiftedTimeStr(startTime, i):
    #headDataMaker内の日付文字列を取得
    td = datetime.timedelta(minutes = i)
    dateT = startTime - td
    res = dateT.strftime("%y/%m/%d %H時%M分")
    res += "00秒"
    return res

def candleVisualize(df):
    fig = plt.figure()
    ax = plt.subplot()

    ohlc = df.values[:, 3:]
    ohlc = np.concatenate([np.arange(ohlc.shape[0]).reshape(-1, 1),\
                           ohlc], axis = 1)

    candlestick_ohlc(ax, ohlc, width=0.7, colorup='g', colordown='r')

    
    
class MainDataMaker():
    START_V = 106.000

    def __init__(self):
        return 
    
    def getMain(self, winLen, changePointList, noise):
        self.winLen = winLen
        self.changePointList= changePointList
        self.noise = noise
        self.direction = random.choice([1, -1])
        
        resList = []
        nowV = self.START_V
    
        for i in range(winLen):
            if(i in self.changePointList):
                self.direction *= -1 
            nextData =  self.getNextMainData(nowV)
            resList.append(nextData)
            nowV = nextData[3]
        resList = np.array(resList)[:, ::-1]
        resDf = pd.DataFrame(resList, columns = COLUMNS_NAME[3:])
        
        return resDf
    

    def getNextMainData(self, nowV):
        #noiseは0~0.02推奨
        openV = nowV + random.uniform(-self.noise, self.noise)

        delta = abs(random.gauss(0, 0.01)) * self.direction
        closeV = nowV + delta

        maxV = max(openV, closeV) + abs(random.gauss(0, 0.005))
        minV = min(openV, closeV) - abs(random.gauss(0, 0.005))

        return [openV, maxV, minV, closeV]
 