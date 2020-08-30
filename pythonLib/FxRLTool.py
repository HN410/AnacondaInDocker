import glob
import os
import random
import gym
import gym.spaces
import numpy as np
from rl.callbacks import Callback
import matplotlib.pyplot as plt
import sys
sys.path.append("/workdir/pythonLib")

from FxDataLoader import *

##状態に建玉の保有状態をふくむか
class FxDealerTrainer(gym.Env):
    SPREAD = 0.002
    def __init__(self, diff = True, nbNotUsedTablediff = True,\
                 prepareWindow = 20, fileGlob = "./minute/*.csv" ,\
                std_sep = StdNormalizer.STD_SEP, stdX = []):
        #self.X        値動きの表　差分をとり正規化済　
        #self.closeList 終値だけ 　生データ
        #self.Xnorm      self.Xを正規化する
        #self.nbList    一続きのself.Xの数　len(self.X), len(self.closeList)に対応 
        #self.stateSize  1stepの状態空間の大きさ　＝　self.X[i][i, :, :]の大きさ
        
        #self.Holding  エージェントが建玉を保有しているか　-1, 0, 1 が売り、なし、買いに対応
        #self.startHoldingTime  エージェントが建玉を保有し始めたとき
        #self.startHoldingValue  ↑の時のcloseの値
        #self.episodeIndex  今エピソードで使われるself.Xのインデックス
        #self.stepIndex    現在のself.X[i]のインデックス
        #self.nowX        現在使われる状態表　＝　self.X[self.episodeIndex]
        #self.nowCloseList 現在使われるCloseList
        #self.nowClose    現在のcloseの値
        #self.nowPositionV 現在の建玉の価値
        
        self.action_space = gym.spaces.Discrete(4) # 行動空間。売る、何もしない　買う　決済　の４つ
        fileList = glob.glob(fileGlob)
        self.dataLoader = FxDataLoader(fileList)
        self.X = self.dataLoader.get_X(1, 1, diff = diff, isForRL = True)
        self.Xnorm = StdNormalizer(self.X[0], std_sep, False, stdX)
        self.X = [self.Xnorm.stdNormalize(i)[:, :, :4] for i in self.X]
        self.nbList = len(self.X)
        self.closeList = self.dataLoader.close_list
        self.stateSize = self.X[0].shape[1:3]
        
        
        high = np.array([3, 3, 3, 3,
#                          3, 6, 3, 6, 10, 10,\
#                         10, 10, 6, 6, 6,\
                        1, float("inf")]) 
        highArray = np.array(high) # 観測空間(state)の最大値
        
        
        self.observation_space = gym.spaces.Box(low=-highArray, high=highArray) # 最小値は、最大値のマイナスがけ
#         self.reward_range = [-1, 1]

    # 各stepごとに呼ばれる
    # actionを受け取り、次のstateとreward、episodeが終了したかどうかを返すように実装
    '''観測空間は、いつもの長さ15のリストと、
    16, 1 買い建玉保有中　0 建玉なし -1 売り建玉
    17, 建玉の価値　正規化済み差分をどんどん足していく
     他のDモデルの使用結果
    '''
    def getNowState(self):
        state = self.nowX[self.stepIndex, :, :].reshape(self.stateSize[-1])
        state = np.append(state, self.Holding)
        state = np.append(state, self.nowPositionV)
        return state

    
    def step(self, action):
        self.nowClose = self.nowCloseList[self.stepIndex]
        # actionを受け取り、次のstateを決定
        acc = (action - 1)
        reward = 0
        #複数の建玉は持てないとする　買いの後に買いをしても意味なし
        if(self.Holding != 0):
            self.nowPositionV  += self.Holding * self.nowX[self.stepIndex, -1, 3]
        
        if(acc != 0):#何か行動したら
            if(self.Holding != 0):# 建玉持ってたら
                if(acc == 2):
                    reward = self.Holding * (self.nowClose - self.startHoldingValue) - FxDealerTrainer.SPREAD
                    reward *= 100
                    self.Holding = 0
                    self.nowPositionV = 0.0
                else:
                    reward = -2
            else:
                if(acc != 2):
                    self.Holding = acc
                    self.startHoldingTime = self.stepIndex
                    self.startHoldingValue = self.nowClose
                    reward = 0.01
                else:
                    reward = -2
        

        self.stepIndex += 1
        done = (self.stepIndex == self.nowX.shape[0])
        if(done):
            self.stepIndex -=1
  
        # 次のstate、reward、終了したかどうか、追加情報の順に返す
        # 追加情報は特にないので空dict
        return  self.getNowState(), reward, done, {}

    # 各episodeの開始時に呼ばれ、初期stateを返すように実装
    def reset(self):
        self.Holding = 0
        self.episodeIndex = random.randint(0, len(self.X)-1)
        self.stepIndex = 0
        self.nowX = self.X[self.episodeIndex]
        self.nowCloseList = self.closeList[self.episodeIndex]
        self.nowPositionV = 0
        
        return self.getNowState() 
    
    def render(self, mode = "human", close = False):
        pass
    

class EpisodeLogger(Callback):
    def __init__(self):
        self.rewards = {}
        self.actions = {}
        self.closes = {}
    def on_episode_begin(self, episode, logs):
        self.rewards[episode] = []
        self.actions[episode] = []
        self.closes[episode] = [0.0]
        self.closeDif = []
    def on_episode_end(self, episode, logs):
        self.closes[episode][0] = self.closeDif[0]
        for i in range(1, len(self.closeDif)):
            self.closes[episode].append(self.closes[episode][-1] +\
                                        self.closeDif[i])
            
    def on_step_end(self, step, logs):
        episode = logs["episode"]
        self.rewards[episode].append(logs["reward"])
        self.actions[episode].append(logs["action"])
        self.closeDif.append(logs["observation"][-1, 3])
        
    def visualize(self, index, xlim = [], ylim = []):
        plt.plot(self.rewards[index], color = "blue", label = "rewards")
        plt.plot(self.actions[index], color = "red", label = "actions")
        plt.plot(self.closes[index], color = "orange", label = "close")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("pos")
        if(xlim):
            plt.xlim(xlim[0], xlim[1])
        if(ylim):
            plt.ylim(ylim[0], ylim[1])
    
    