#l 56  *2を外す
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
USED_INDEX_N = 4

##状態に建玉の保有状態をふくむか
class FxDealerTrainer(gym.Env):
    SPREAD = 0.006
    
    def __init__(self, diff = True, nbNotUsedTablediff = True,\
                 prepareWindow = 20, fileGlob = "./minute/*.csv" ,\
                std_sep = StdNormalizer.STD_SEP, stdX = [],\
                 used_index_n = 4, isForMakingTeacherData = False):
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
        #self.countTime  建玉を保有し続けた時間
        #self.countNoMoveTime 何もしていない時間
        #isForMakingTeacherData 教師あり学習に使うとき用
        
        self.action_space = gym.spaces.Discrete(4) # 行動空間。売る、何もしない　買う　決済　の４つ
        fileList = glob.glob(fileGlob)
        self.dataLoader = FxDataLoader(fileList)
        self.X = self.dataLoader.get_X(1, 1, diff = diff, isForRL = True)
        self.Xnorm = StdNormalizer(self.X[0], std_sep, False, stdX)
        self.X = [self.Xnorm.stdNormalize(i)[:, :, :used_index_n] for i in self.X]
        self.nbList = len(self.X)
        self.closeList = self.dataLoader.close_list
        self.stateSize = self.X[0].shape[1:3]
        self.countTime = 0
        self.countNoMoveTime = 0
        self.countSettleTime = 0
        self.isForMakingTeacherData = isForMakingTeacherData
        
        
        high = np.array([3, 3, 3, 3,
                          3, 6, 3, 6, 10, 10,\
                         10, 10, 6, 6, 6])[:used_index_n] * 2
        added_data = np.array([1, float("inf")]) 
        high  = np.concatenate([high, added_data])
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
    
    def getAddedDict(self, pureReward):
        result = {}
        if(self.isForMakingTeacherData):
            result["pureReward"] = pureReward
        return result
            
    # 各episodeの開始時に呼ばれ、初期stateを返すように実装
    def step(self, action):
        self.nowClose = self.nowCloseList[self.stepIndex]
        # actionを受け取り、次のstateを決定
        acc = (action - 1)
        reward = 0
        pureReward = 0
        #複数の建玉は持てないとする　買いの後に買いをしても意味なし
        if(self.Holding != 0):
            self.nowPositionV  += self.Holding * self.nowX[self.stepIndex, -1, 3] / 10
            self.countTime += 1
            #大きな負の建玉は持たない
            if(self.nowPositionV < -3):
                reward += self.nowPositionV *0.1
            elif(self.nowPositionV < -2):
                reward += self.nowPositionV *0.1
            #長時間建玉を持たない
            if(self.countTime > 30):
                reward -= self.countTime * 0.001
            elif(acc == 0):
                reward += 0.01
        
        if(acc != 0):#何か行動したら
            self.countNoMoveTime = 0
            if(self.Holding != 0):# 建玉持ってたら
                if(acc == 2):
                    reward = self.Holding * (self.nowClose - self.startHoldingValue) - FxDealerTrainer.SPREAD
                    reward *= 10
                    pureReward = reward
                    if(self.Holding * (self.nowClose - self.startHoldingValue) > 0):
                        reward += 0.001

                    self.Holding = 0
                    self.nowPositionV = 0.0
                    self.countTime = 0
                    self.countSettleTime += 1
                    if(self.countTime > 20):
                        reward += 0.1
                    else:
                        reward += 0.005 * self.countTime
                    
                else:
                    reward -= 0.2
            else:
                if(acc != 2):
                    self.Holding = acc
                    self.startHoldingTime = self.stepIndex
                    self.startHoldingValue = self.nowClose

                else:
                    self.countNoMoveTime += 1
                    if(self.countNoMoveTime > 20):
                        reward -= 0.001 * self.countNoMoveTime
                    reward -= 0.2
                
                    
        elif(self.Holding == 0):
            self.countNoMoveTime += 1
            if(self.countNoMoveTime > 20):
                reward -= 0.001 * self.countNoMoveTime
        else:
            if(self.Holding * (self.nowClose - self.startHoldingValue) > 0):
                reward += 0.2 / (self.countTime ** (1/2))
                
        self.stepIndex += 1
        done = (self.stepIndex == self.nowX.shape[0])
        if(done):
            self.stepIndex -=1
            
        addedDict = self.getAddedDict(pureReward) 
  
        # 次のstate、reward、終了したかどうか、追加情報の順に返す
        # 追加情報は特にないので空dict
        return  self.getNowState(), reward, done, addedDict

    def resetDefineIndex(self, episodeIndex):
        self.Holding = 0
        self.episodeIndex = episodeIndex
        self.stepIndex = 0
        self.nowX = self.X[self.episodeIndex]
        self.nowCloseList = self.closeList[self.episodeIndex]
        self.nowPositionV = 0
        self.countTime = 0
        self.countNoMoveTime = 0
        self.countSettleTime = 0
        return self.getNowState()
        
    
    def reset(self):
        result = self.resetDefineIndex(random.randint(0, len(self.X)-1))
        return result
    
    def render(self, mode = "human", close = False):
        pass
    

class EpisodeLogger(Callback):
    def __init__(self):
        self.rewards = {}
        self.pureRewards = {}
        self.actions = {}
        self.closes = {}
        
    def on_episode_begin(self, episode, logs):
        self.rewards[episode] = []
        self.pureRewards[episode] = []
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
        
        #logs["info"]["pureReward"]で返される値が0-dimensionなのでスカラーにするため０を足す
        self.pureRewards[episode].append(logs["info"]["pureReward"] + 0)
        
        self.actions[episode].append(logs["action"])
        self.closeDif.append(logs["observation"][ 3])
    
    def getPureRewardSum(self, episode):
        return sum(self.pureRewards[episode])

        
    def visualize(self, index, xlim = [], ylim = [], isPureReward = False):
        if(isPureReward):
            plt.plot(self.rewards[index], color = "blue", label = "rewards")
        else:
            plt.plot(self.pureRewards[index], color = "blue", label = "pureRewards")            
        plt.plot(self.actions[index], color = "red", label = "actions")
        plt.plot(self.closes[index], color = "orange", label = "close")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("pos")
        if(xlim):
            plt.xlim(xlim[0], xlim[1])
        if(ylim):
            plt.ylim(ylim[0], ylim[1])
    
    