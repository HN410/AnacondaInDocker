import glob
import os
import random
import gym
import gym.spaces
import numpy as np
import sys
import pandas as pd
from tensorflow.keras.models import load_model


sys.path.append("/workdir/pythonLib")
from machine_learning_tool import *
from FxRLTool import *
from FxDataLoader import *
POSITION_UPPER_LIM = 2
POSITION_LOWER_LIM = -0.5 


def recordXY(env, i):
    resultX = []
    resultY = []
    rewardList = []
    
    stateArray = []
    stateArray.append(env.resetDefineIndex(i))
    for i in range(99):
        nowState, nowReward, nowDone, nowAddedD =  env.step(1)
        stateArray.append(nowState)
    stateArray = np.array(stateArray)
    
    isDone = False
    pureReward = 0
    while(not isDone):
        resultX.append(stateArray)
        acc = getNextAction(stateArray, pureReward, env.countTime)
        
        nowState, nowReward, isDone, nowAddedD = env.step(acc)
        
        stateArray = np.concatenate([stateArray[1:, :],\
                                     nowState.reshape(1, -1)], axis = 0)
        pureReward = nowAddedD["pureReward"]
        rewardList.append(nowReward)
        resultY.append(acc) 
    return np.array(resultX), np.array(resultY), np.array(rewardList)
        
        
def getNextAction(stateArray, pureReward, countTime):
    bState = stateArray[-1, :]
    bHasPosition = bState[-2]
    bPositionV = bState[-1]
    
    if(bHasPosition != 0):
        if(countTime > 30):
            return 3
        elif(countTime > 20 and bHasPosition * bState[3] < 0):
            return 3
        elif(countTime > 10 and bPositionV > POSITION_UPPER_LIM / 2):
            return 3
        if(bPositionV > POSITION_UPPER_LIM or bPositionV < POSITION_LOWER_LIM):
            return 3
        return 1
    else:
        deepLData = getDeepLData(stateArray)
        if(stateArray[-2, -2] == 0 ):
            closeMin = np.min(stateArray[-3:, 5])
            closeMax = np.max(stateArray[-3:, 5])
            if(closeMin > 0):
                return 2
            elif(closeMax < 0):
                return 0
        return 1

def getDeepLData(stateArray):
    return
    
    
    
import matplotlib.pyplot as plt

def visualize(X, Y, reward, xlim = [], ylim = []):
    close = [0]
    for i in range(1, X.shape[0]):
        close.append(close[i-1] + X[i-1, -1, 3])
    plt.plot(reward, color = "blue", label = "rewards")
    plt.plot(Y, color = "red", label = "actions")
    plt.plot(close, color = "orange", label = "close")
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("pos")
    if(xlim):
        plt.xlim(xlim[0], xlim[1])
    if(ylim):
        plt.ylim(ylim[0], ylim[1])   
