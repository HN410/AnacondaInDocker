import pandas as pd
import numpy as np
import math
from datetime import datetime as dt
import datetime

class StdNormalizer():
    STD_SEP = [13, 14, 15]
    CLOSE_N = 3
    

    def __init__(self, inputs, sep_list, isY = False,  std_list = []):
        self.std_list = []
        self.sep_list = sep_list
        
        if(std_list != []):
            self.std_list = std_list
        else:
            if(isY):
                self.std_list.append(np.std(inputs))
            else:
                i = 0
                for j in sep_list:
                    self.std_list.append(np.std(inputs[:, :, i:j]))
                    i = j

        
    def stdNormalize(self, inputs):
        i = 0
        for n, j in enumerate(self.sep_list):
            inputs[:, :, i:j] = (inputs[:, :, i:j]) / self.std_list[n]
            i = j
        return inputs
     
    def stdNormalizeY(self, inputs, ind = 0):
        inputs = inputs / self.std_list[ind]
        return inputs 



class FxDataLoader:
    BINARY_CATEGORY_N = 3
    CLOSE_N = 3
    CLOSE_DIFF_LABEL_N = 12

    
    def __init__(self, data_paths, label_list = ["始値(Bid)", "高値(Bid)", "安値(Bid)", "終値(Bid)"],\
                date_label = "日付", target_label = "終値(Bid)",\
                drop_list = ["Date", "Month", "Day"]) :
        self.market_infos = []
        for i in data_paths:
            self.market_infos.append(pd.read_csv(i))
        self.label_list = label_list
        self.ema12s = []
        self.processed_datas = []
        label_before = label_list + ["ma5", "ma25", "ema5", "ema25",\
        "b_band_u1", "b_band_u2", "b_band_d1", "b_band_d2","range", "macd", "macd_s_processed"]
        label_after = label_before.copy()
        self.label_after = label_after
        label_after[0:4] =["Open", "High", "Low", "Close"]
        transTable = str.maketrans({"時":":", "分":":", "秒":""})

        
        for info in self.market_infos:
            info = info.sort_index(ascending = False).reset_index().drop(columns = "index")
            for i in label_list:
                info[i] = info[i].astype("float32")
                info = info.sort_index(ascending = False)
                info = info.reset_index().loc[:, ["index", date_label]+label_list]
            numpy_close = info.loc[:, target_label].values 
            ma5 = self.MA(numpy_close, 5)
            ma25 = self.MA(numpy_close, 25)
            info["ma5"] = ma5
            info["ma25"] = ma25

            ema5 = self.EMA(numpy_close, 5)
            ema12 = self.EMA(numpy_close, 12)
            self.ema12s.append(ema12)
            ema25 = self.EMA(numpy_close, 25)
            info["ema5"] = ema5
            info["ema25"] = ema25

            b_b = np.array(self.b_band(numpy_close, 20))
            ma20 = np.array(self.MA(numpy_close, 20))
            b_band_u1 = ma20 + b_b
            b_band_u2 = b_band_u1 + b_b
            b_band_d1 = ma20 - b_b
            b_band_d2 = ma20 - 2 * b_b
            info["b_band_u1"] = b_band_u1
            info["b_band_u2"] = b_band_u2
            info["b_band_d1"] = b_band_d1
            info["b_band_d2"] = b_band_d2

            kwargs = {"range" : lambda x: (x["終値(Bid)"]-x["始値(Bid)"]) }
            info = info.assign(**kwargs)

            macd = self.MACD(numpy_close)
            info["macd"] = macd
            info["macd_s_processed"] = macd - np.array(self.MACD_Signal(macd))

            model_data = info
 
            processed_values = model_data.loc[:, label_before].values
            processed_data = pd.DataFrame(data = processed_values,  columns = label_after)
            processed_data["Date"] = np.reshape(model_data.loc[:, ["日付"]].values, (-1))

            date_list = processed_data.loc[:, "Date"].values.tolist()
            date_list = [dt.strptime(i.translate(transTable),\
                                                "%y/%m/%d %X") for i in date_list]
            processed_data["Month"] = [i.month for i in date_list]
            processed_data["Day"] = [i.day for i in date_list]
            processed_data["WeekDay"] = [i.weekday() for i in date_list]
            processed_data["Hour"] = [i.hour for i in date_list] 

            for i in drop_list:
                processed_data = processed_data.drop(i, 1)
            self.processed_datas.append(processed_data)

    
    def MA(self, data, n):
        result = []
        for i in range(n-1):
            result.append(data[i])
        for i in range(len(data)-n+1):
            result.append(float(sum((data[i:i+n])))/n)
        return result
    def EA(self, list, n):
        if(len(list) == 1):
            return list[0]
        elif(len(list) == 2):
            return list[0] + 2 * (list[1] - list[0])/(n+1)
    def EMA(self, data, n):
        result = []
        result.append(data[0])
        for i in range(1, len(data)):
            result.append(self.EA([result[i-1], data[i]], n))
        return result
    def b_band(self, data, n=20):
        squared_list = [i**2 for i in data]
        result = []
        result += [0] * (n-1)

        for i in range(len(data)- n+1):
            value = float(n * (sum(squared_list[i:i+n])) - sum(data[i:i+n])**2)
            value /= n*(n-1)
            value = math.sqrt(value)
            result.append(value)
        return result
    def MACD(self, data, n1=12, n2=25):
        ema1 = self.EMA(data, n1)
        ema2 = self.EMA(data, n2)
        return np.array(ema1)-np.array(ema2)
    def MACD_Signal(self, macd, n = 9):
        return self.EMA(macd, n)
    
    def get_processed_datas(self):
        return self.processed_datas
    
    def get_X(self, window, y_window, diff = False, prepare_window = 25,  isForRL = False):
        if(isForRL):
            self.X = []
        else:
            self.X = np.array([])
        self.close_list = []
        for i, data in enumerate(self.processed_datas):
            data_v = data.values
            list_ = []
            base_v = data_v[:, FxDataLoader.CLOSE_N].copy()
            base_list = base_v[prepare_window+ window:len(data_v) - y_window + 1]
            self.close_list.append(base_list.copy())
            if(diff):
                data_v[:, :FxDataLoader.CLOSE_N] = data_v[:, :FxDataLoader.CLOSE_N]-base_v.reshape(-1, 1)
                data_v[:, FxDataLoader.CLOSE_N + 1:FxDataLoader.CLOSE_DIFF_LABEL_N] =\
                data_v[:, FxDataLoader.CLOSE_N + 1:FxDataLoader.CLOSE_DIFF_LABEL_N]-base_v.reshape(-1, 1)
                data_v[1:, 3] = data_v[1:, 3] - base_v[:-1]
                
            for val in range(len(data_v) - window - y_window -prepare_window  + 1):
                item = data_v[prepare_window + val : prepare_window + val + window, :]
                ## if(diff):  
                    ## item[0, 3] = 0 はじめのcloseの値を０に基準化b_band_d2まで
                    ## item[:, :len(self.label_after)-3] = item[:, :len(self.label_after)-3] - base   
                list_.append(item)

            list_ = np.array(list_)
            if(isForRL):
                self.X.append(list_)
            else:
                if (list_.shape[0] != 0):
                    if i == 0:
                        self.X = list_.copy()
                        self.base_X = base_list.copy()
                    else :
    ##                    print(self.X.shape, list.shape)
                        self.X = np.r_[self.X, list_]
                        self.base_X = np.r_[self.base_X,  base_list.copy()]
        if(not isForRL):
            self.base_X = np.array(self.base_X)
        return self.X
    
    def get_Y(self, window, y_window, type = "value", close_n = 3, limit = 0.03, prepare_window = 25):
        self.Y = np.array([])
        for i, data in enumerate(self.processed_datas):
            data_v = data.values
            list = []
            for j in range(len(data_v) - window - y_window - prepare_window +1):
                value = 0
                lot = data_v[prepare_window + j + window -1: prepare_window + j + window + y_window, close_n].reshape(-1)
                if(type == "value"):
                    value = lot[-1]
                elif(type == "updown1"):
                    value = self.get_Y_updown1(lot)
                elif(type == "updown2" ):
                    value = self.get_Y_updown2(lot)
                elif(type == "updown3"):
                    value = self.get_Y_updown3(lot, limit)
                list.append(value)
            list = np.array(list)
            
            if i == 0:
                self.Y = list.copy()
            else :
                self.Y = np.r_[self.Y, list]
        if (type == "value"):
            self.Y = self.Y - self.base_X
        return self.Y
    
    def get_Y_updown1(self, lot):
        size = lot.shape[0]
        list = []
        base = lot[0]
        for i in range(size // 5):
            val = (lot[i * 5] - base )* (2 ** i)
            list.append(val)
        return sum(list)
    
    def get_Y_updown2(self, lot):
        list = []
        base = lot[0]
        return max(lot) + min(lot) - base*2
    
    def get_Y_updown3(self, lot, limit):
        base = lot[0]
        index = lot.shape[0]-1
        for val in range(lot.shape[0]):
            if(abs(lot[val] - base) >  limit):
                index = val
                break
        return lot[index] - base


    def get_Y_binary(self, Y, limit):
        result = np.ones(Y.shape, dtype = "int8")
        result = np.where(Y > limit, 2, result)
        result = np.where(Y < -limit, 0, result)
        for i in range(FxDataLoader.BINARY_CATEGORY_N):
            print("number of index " + str(i) + " : " +\
                   str(len(result[result == i]))) 

        return result
    
    def get_X_analyze(self, window, back = 0, diff = False):
        data  = self.processed_datas[-1]
        data_v = data.values
        item = data_v[data_v.shape[0]- window - back-1 :data_v.shape[0] - back-1, :].copy()
        base_v = item[:, FxDataLoader.CLOSE_N]
        if(diff):
            item[:, :FxDataLoader.CLOSE_N] = item[:, :FxDataLoader.CLOSE_N]-base_v.reshape(-1, 1)
            item[:, FxDataLoader.CLOSE_N + 1:FxDataLoader.CLOSE_DIFF_LABEL_N] =\
            item[:, FxDataLoader.CLOSE_N + 1:FxDataLoader.CLOSE_DIFF_LABEL_N]-base_v.reshape(-1, 1)
            item[1:, FxDataLoader.CLOSE_N ] = item[1:, FxDataLoader.CLOSE_N ] - base_v[:-1]
            item[0, FxDataLoader.CLOSE_N] = 0
        self.X_analyze = item
        base = base_v[-1]

        return item, base
    
    def add_data(self, lis, dt,  processed_data_ind = 0):
        #open high low close
        values = self.processed_datas[0].values
        closes = values[:, 3]
        length = values.shape[0]
        
        ma5 = float(sum(closes[length-4:] + lis[3]) / 5)
        ma25 = float(sum(closes[length-24:] + lis[3]) / 25)
        
        ema5 = values[6, -1] + 2 * (lis[3] - values[6, -1]) / 6
        ema25 = values[7, -1] + 2 * (lis[3] - values[7, -1]) / 26
        
        closes20 = np.append(closes[length-19:], lis[3])
        squared_list = [i**2 for i in closes20]
        value = float(20 * (sum(squared_list)) - sum(closes20)**2)
        value /= 20 * 19
        value = math.sqrt(value)
        b_b = value
        
        ma20 = sum(closes[length-19:] + lis[3]) / 20
        b_band_u1 = ma20 + b_b
        b_band_u2 = b_band_u1 + b_b
        b_band_d1 = ma20 - b_b
        b_band_d2 = ma20 - 2 * b_b
        
        range_ = lis[3] - lis[0]
        
        ema12 = self.ema12s[0][-1] + 2 * (lis[3] - self.ema12s[0][-1]) /13
        self.ema12s[0].append(ema12)
        macd = ema12 - ema25
        
        sig_b = values[13, -1] - values[14, -1] 
        macd_sig = sig_b + 2 * (macd - sig_b) / 10
        macd_sig = macd - macd_sig
        
        result = lis + [ma5, ma25, ema5, ema25, b_band_u1, b_band_u2,\
                        b_band_d1, b_band_d2, range_, macd, macd_sig, dt.weekday(), dt.hour]
        self.processed_datas[0].append(pd.Series(result, columns = self.processed_datas[0].columns, ignore_index = True))
        
        
        return self.processed_datas[0]

    def get_learning_data(self, window_l , y_window, split_index, type = "value",\
                          binary = False, binary_limit = 1, close_n = StdNormalizer.CLOSE_N, limit = 0.03, prepare_window = 25,\
                          diff = True , std_sep = StdNormalizer.STD_SEP, ans = False, stdX = [],stdY = []):
        X_training = []
        X_test = []
        Y_training = []
        Y_test = []
        for n, window in enumerate(window_l):
            X = self.get_X(window, y_window, diff, prepare_window)
            Y = self.get_Y(window, y_window, type, close_n, limit, prepare_window)

            if(not binary):
                if(n==0):
                    if(stdY == []):                        
                        self.Ynorm = StdNormalizer(Y, [-1], True)
                        print("Y stddev: " + str(self.Ynorm.std_list))
                    else:
                        self.Ynorm = StdNormalizer(Y, [-1], True, stdY)
                Y = self.Ynorm.stdNormalizeY(Y)

            inputs = X[:, :, :15]
            if(n ==0):
                if(stdX == []):
                    self.Xnorm = StdNormalizer(inputs, std_sep)
                    print("X stddev:" + str(self.Xnorm.std_list))
                else:
                    self.Xnorm = StdNormalizer(inputs, std_sep, False, stdX)
            inputs = self.Xnorm.stdNormalize(inputs)

            while(True):
                if(binary):
                    outputs = self.get_Y_binary(Y, binary_limit)
                else:
                    outputs = Y

                training_inputs = inputs[split_index:]
                test_inputs = inputs[:split_index]
                training_outputs = outputs[split_index:]
                test_outputs = outputs[:split_index] 
                flag = False
                if(binary and n ==0 and not ans):
                    for i in range(FxDataLoader.BINARY_CATEGORY_N):
                        np.count_nonzero(test_outputs == i)
                    while(True):
                        t = input("これで続ける場合はyを、それ以外は新たなbinary_limitを" +\
                             "入力してください")
                        if (t != "y"):
                            try:
                                binary_limit = float(t)
                                flag = True
                            except ValueError:
                                t = input("数を入力してください")
                                continue
                        break
                if(not flag):
                    break

            X_training.append(training_inputs)
            X_test.append(test_inputs)
            Y_training.append(training_outputs)
            Y_test.append(test_outputs)

            if n == 0:
                split_index += max(window_l) - min(window_l) + 10
        return X_training, Y_training, X_test, Y_test
    
            
