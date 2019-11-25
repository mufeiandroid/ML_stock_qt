# -*- coding:utf8 -*-
import pandas as pd
import talib 
# import ta
import numpy as np

from scipy.stats import rankdata
import scipy as sp

from sklearn import preprocessing
abs_scaler = preprocessing.MaxAbsScaler()
import copy


def sigmoid(X,useStatus):
    if useStatus:
        return 1.0 / (1 + np.exp((0.001*X)));
    else:
        return (X)
    
    
# TSRANK 函数
def func_rank(na):
    return rankdata(na)[-1]/rankdata(na).max()


# DECAYLINEAR函数
def func_decaylinear(na):
    n = len(na)
    decay_weights = np.arange(1,n+1,1) 
    decay_weights = decay_weights / decay_weights.sum()

    return (na * decay_weights).sum()


# HIGHDAY 函数
def func_highday(na):
    return len(na) - na.argmax()


# LOWDAY 函数 
def func_lowday(na):
    return len(na) - na.argmin()


def cut_to1(alpha,minx=-1,maxx=1):
    alpha=alpha.copy()
    alpha[alpha>maxx]=maxx 
    alpha[alpha<minx]=minx 
    return alpha


def arbr(df):
    df['HO']=df.high-df.open
    df['OL']=df.open-df.low
    df['HCY']=df.high-df.close.shift()
    df['CYL']=df.close.shift()-df.low
    df['MID']=0.5*(df.high+df.low)
    df['uppwer']=df.high-df['MID']
    df['downpwer']=df['MID']-df.low
    #计算AR、BR指标
    df['AR']=talib.SUM(df.HO, timeperiod=26)/(1+talib.SUM(df.OL, timeperiod=26))-1
    df['BR']=talib.SUM(df.HCY, timeperiod=26)/(1+talib.SUM(df.CYL, timeperiod=26))-1
    df['CR']=talib.SUM(df.uppwer, timeperiod=26)-talib.SUM(df.downpwer, timeperiod=26)-1

    return df['AR'],df['BR'],df['BR']


def CCI(data, ndays): 
    TP = (data['high'] + data['low'] + data['close']) / 3 
    CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.015 * TP.rolling(ndays).std()),
    name = 'CCI') 
    data = data.join(CCI) 
    return data


def EVM(data, ndays): 
    print(data['high'] + data['low'])
    dm = ((data['high'] + data['low'])) - ((data['high'].shift(1) + data['low'].shift(1)))
    r=data['high'] - data['low']
    r[r==0]=0.0001
    br = (data['volume'] ) / r
    EVM = 0.5*dm / br 
    EVM_MA = -pd.Series(EVM.rolling(ndays).mean(), name = 'EVM') 
    data = data.join(EVM_MA) 
    return data


def ForceIndex(data, ndays):
    FI = pd.Series(data['close'].diff(ndays) * data['volume'], name = 'ForceIndex') 
    data = data.join(FI) 
    return data


def my_ROC(data,n):
    N = data['close'].diff(n)
    D = data['close'].shift(n)
    ROC = pd.Series(N/D,name='myRoc')
    data = data.join(ROC)
    return data 


def sunup(df):
    df['sunup']=1*(df['close']>df['open'])
    return df['sunup']


def upratio(df,days):
    df['sunup']=sunup(df)
    upratio=df['sunup'].rolling(days).sum()/days
    return upratio


def trans_features(stockk):  
    """该函数主要用于继续新增各种个股的特征，增加了各种常用技术指标入EMA，布林线等"""
    # stockk就是融合了各种特征列，标签列的df格式的data数据
    # stockk[['open','close','low','high']]=stockk[['open','close','low','high']]/stockk['close'].iloc[-1]
    
    # 特征工程，继续增加新特征   
    stockk['preclose']=stockk['close'].shift()
    stockk['volume']=stockk['volume']/1000000  # 个股交易量数据缩放
    stockk['amount']=stockk['amount']/1000000
    stockk['pctChg']=cut_to1(8*stockk['close'].diff()/stockk['close'].shift())  # 当日涨幅
    # stockk['pctChg2']=(1+stockk['pctChg']).rolling(2).cumprod()-1
    stockk['pctChg2']=cut_to1(5*(stockk['close']/stockk['close'].shift(2))-5)  # 过去2日涨幅
    stockk['pctChg4']=cut_to1(4*(stockk['close']/stockk['close'].shift(5))-4)
    stockk['pctChg8']=cut_to1(2*(stockk['close']/stockk['close'].shift(8))-2)
    stockk['pctChg15']=cut_to1(2*(stockk['close']/stockk['close'].shift(15))-2)
    stockk['pctChg30']=cut_to1(2*(stockk['close']/stockk['close'].shift(15))-2)
    stockk['avg_price']=stockk['amount']/(0.01+stockk['volume'])  # 平均成交价，成交额/交易量

    stockk['sma5']=stockk['close'].rolling(5).mean()  # 个股前5日均价
    stockk['sma10']=stockk['close'].rolling(10).mean()
    stockk['sma20']=stockk['close'].rolling(20).mean()
    stockk['sma40']=stockk['close'].rolling(40).mean()
    stockk['sma80']=stockk['close'].rolling(80).mean()

    stockk['max20']=stockk['high'].rolling(20).max().shift()  # 个股前20日最高价(不含当日)
    stockk['max60']=stockk['high'].rolling(60).max().shift()
    stockk['max120']=stockk['high'].rolling(120).max().shift()

    stockk['min20']=stockk['low'].rolling(20).min().shift()  # 个股前20日最低价(不含当日)
    stockk['min60']=stockk['low'].rolling(60).min().shift()
    stockk['min120']=stockk['low'].rolling(120).min().shift()

    stockk['20_min']=cut_to1(1*stockk['close']/stockk['min20']-0.9)### (个股20日内最大涨幅 + 10%)，涨幅超110%，则截断。----10%有何用？？？？？？
    stockk['60_min']=cut_to1(1*stockk['close']/stockk['min60']-0.9)###
    stockk['120_min']=cut_to1(1*stockk['close']/stockk['min120']-0.9)###

    stockk['20_max']=cut_to1(2*stockk['close']/stockk['max20']-2)### 2*(个股20日内最大跌幅), 跌幅超50%，则截断。
    stockk['60_max']=cut_to1(2*stockk['close']/stockk['max60']-2)### （即便昨天是max20，今天最多涨10%，不会截断）
    stockk['120_max']=cut_to1(2*stockk['close']/stockk['max120']-2)###

    stockk['5_d']=cut_to1(20*stockk['sma5'].diff()/stockk['sma5'].shift())  # 个股5日均线涨幅，±5%截断
    stockk['10_d']=cut_to1(40*stockk['sma10'].diff()/stockk['sma10'].shift())
    stockk['20_d']=cut_to1(60*stockk['sma20'].diff()/stockk['sma20'].shift())
    stockk['40_d']=cut_to1(80*stockk['sma40'].diff()/stockk['sma40'].shift())
    stockk['80_d']=cut_to1(100*stockk['sma80'].diff()/stockk['sma80'].shift())

    stockk['r5']=cut_to1(6*stockk['close']/stockk['sma5']-6)  # 个股收盘价较5日均线偏移幅度（±16.6%截断）
    stockk['r10']=cut_to1(4*stockk['close']/stockk['sma10']-4)
    stockk['r20']=cut_to1(4*stockk['close']/stockk['sma20']-4)
    stockk['r40']=cut_to1(3*stockk['close']/stockk['sma40']-3)
    stockk['r80']=cut_to1(2*stockk['close']/stockk['sma80']-2)
    
    stockk['r5_20']=(3*stockk['sma5']/stockk['sma20']-3)  # 个股5日均线较20日均线偏移幅度(下面语句会进行±33.3%截断)
    stockk['r10_40']=(2*stockk['sma10']/stockk['sma40']-2)
    stockk['r20_80']=(2*stockk['sma20']/stockk['sma80']-2)
    stockk['r5_40']=(2*stockk['sma5']/stockk['sma40']-2)
    
    stockk['r5_20_d']=cut_to1(10*stockk['r5_20'].diff())  # 5日20日均线偏移幅度变动量（±10%截断）----不太可能一天变动10%吧！
    stockk['r10_40_d']=cut_to1(20*stockk['r10_40'].diff())
    stockk['r20_80_d']=cut_to1(20*stockk['r20_80'].diff())
    
    stockk['r5_20']=cut_to1(stockk['r5_20'])  # ±33.3%截断
    stockk['r10_40']=cut_to1(stockk['r10_40'])
    stockk['r20_80']=cut_to1(stockk['r20_80'])
    stockk['r5_40']=cut_to1(stockk['r5_40'])


    stockk['day_price'] = 0.5*(stockk['open']+stockk['close'])/stockk['volume']  # (开盘价-收盘价)的平均值除以交易量

    stockk['v5']=(10*stockk['volume']/(0.1+stockk['volume'].rolling(5).mean()))  # 交易量/5日交易量均值，0.1是避免出现0交易量，导致报错
    stockk['v10']=(10*stockk['volume']/(0.1+stockk['volume'].rolling(10).mean()))
    stockk['v20']=(10*stockk['volume']/(0.1+stockk['volume'].rolling(20).mean()))
    stockk['v40']=(10*stockk['volume']/(0.1+stockk['volume'].rolling(40).mean()))
    stockk['v80']=(10*stockk['volume']/(0.1+stockk['volume'].rolling(80).mean()))
    
    stockk['v3_9']=(10*stockk['volume'].rolling(3).mean()/(0.1+stockk['volume'].rolling(9).mean()))  # 交易量3日均/交易量9日均
    stockk['v5_20']=(10*stockk['volume'].rolling(5).mean()/(0.1+stockk['volume'].rolling(20).mean()))
    stockk['v9_50']=(10*stockk['volume'].rolling(9).mean()/(0.1+stockk['volume'].rolling(50).mean()))
    
    stockk['v5_d']=0.6366*np.arctan(0.1*stockk['v5'].diff())  # 用arctan()进行数据归一化，交易量过大，用0.1压一压，避难arctan()饱和
    stockk['v10_d']=0.6366*np.arctan(0.1*stockk['v10'].diff())
    stockk['v20_d']=0.6366*np.arctan(0.1*stockk['v20'].diff())
    stockk['v40_d']=0.6366*np.arctan(0.1*stockk['v40'].diff())
    stockk['v80_d']=0.6366*np.arctan(0.1*stockk['v80'].diff())
    
    stockk['v3_9_d']=0.6366*np.arctan(1*stockk['v3_9'].diff())
    stockk['v5_20_d']=0.6366*np.arctan(1*stockk['v5_20'].diff())
    stockk['v9_50_d']=0.6366*np.arctan(1*stockk['v9_50'].diff())
    
    stockk['v3_9']=0.6366*np.arctan(0.1*stockk['v3_9'])
    stockk['v5_20']=0.6366*np.arctan(0.1*stockk['v5_20'])
    stockk['v9_50']=0.6366*np.arctan(0.1*stockk['v9_50'])
    
    stockk['v5']=0.6366*np.arctan(0.1*stockk['v5'])
    stockk['v10']=0.6366*np.arctan(0.1*stockk['v10'])
    stockk['v20']=0.6366*np.arctan(0.1*stockk['v20'])
    stockk['v40']=0.6366*np.arctan(0.1*stockk['v40'])
    stockk['v80']=0.6366*np.arctan(0.1*stockk['v80'])

    # 个股macd(12,26,9)
    stockk['dif'], stockk['dea'],stockk['hist'] = talib.MACD(stockk['close'].astype(float).values, fastperiod=12, slowperiod=26, signalperiod=9)
    stockk['dif']=cut_to1(0.7*stockk['dif'])  # 个股dif截断怎么用1.42吗，超过1.42的dif就没了？？？
    stockk['dea']=cut_to1(0.7*stockk['dea'])
    stockk['hist']=(1*stockk['hist'])

    stockk['hist_d']=cut_to1(4*stockk['hist'].diff())
    stockk['hist']=cut_to1(stockk['hist'])


    #计算几个指标相对前收盘的涨幅
    stockk['highratio']=cut_to1(10*(stockk['high'])/stockk['close'].shift()-10.2) # 当日最高涨幅（为何设10.2？？？）
    stockk['lowratio']=cut_to1(10*(stockk['low'])/stockk['close'].shift()-9.8)
    stockk['openratio']=cut_to1(20*(stockk['open'])/stockk['close'].shift()-20) # 当日开盘涨幅（±5%截断，不合适吧，开盘涨停的没了？？？）

    #talib均线指标
    stockk['tema']=0.6366*np.arctan(0.1*talib.TEMA(stockk['close'],timeperiod=30)**0.5)  # 三重移动平均线

    stockk['trima']=0.6366*np.arctan(0.1*talib.TRIMA(stockk['close'],timeperiod=30)**0.5)  # 三角移动平均线

    stockk['wma']=0.6366*np.arctan(0.1*talib.WMA(stockk['close'],timeperiod=30)**0.5)  # 加权移动平均线
    
    stockk['t3']=0.6366*np.arctan(0.1*talib.T3(stockk['close'],timeperiod=5, vfactor=0)**0.5)  # 超短线使用的三重移动平均线
     
    #talib量价指标
    stockk['natr']=cut_to1(-0.2+0.1*talib.NATR(stockk['high'],stockk['low'],stockk['close'],timeperiod=14))
    stockk['adline'] = 0.6366*np.arctan(0.005*talib.AD(stockk['high'],stockk['low'],stockk['close'],stockk['volume'])-0.5)
    stockk['adosc'] = 0.6366*np.arctan(0.1*talib.ADOSC(stockk['high'],stockk['low'],stockk['close'],stockk['volume'], fastperiod=3, slowperiod=10))
    stockk['obv'] =0.6366*np.arctan(0.001*talib.OBV(stockk['close'],stockk['volume']))
    
    ##静态指标
    stockk['beta'] = 0.6366*np.arctan(talib.BETA(stockk['high'],stockk['low'], timeperiod=5))
    stockk['correl'] = cut_to1(-3*talib.CORREL(stockk['high'],stockk['low'], timeperiod=30)+3)
    stockk['linearreg'] = 0.6366*np.arctan(0.1*talib.LINEARREG(stockk['close'], timeperiod=14)-1)
    stockk['lineangle'] = 0.6366*np.arctan(0.1*talib.LINEARREG_ANGLE(stockk['close'], timeperiod=14))
    # stockk['lineslope'] =0.1*talib.LINEARREG_SLOPE(stockk['close'],  timeperiod=14)
    stockk['var'] =  0.6366*np.arctan(30*talib.VAR(stockk['close'],  timeperiod=5,nbdev=1))
    stockk['TSF'] =  0.6366*np.arctan(2*talib.TSF(stockk['close'], timeperiod=14))
    stockk['lineintercept'] = 0.6366*np.arctan(0.05*talib.LINEARREG_INTERCEPT(stockk['close'], timeperiod=14))
    stockk['stddev'] =cut_to1((talib.STDDEV(stockk['close'], timeperiod=5,nbdev=1))**0.25-1)

    ###动量指标
    stockk['sar']  = 0.6366*np.arctan(0.1*talib.SAR(stockk['high'], stockk['low'], acceleration=0, maximum=0))
    stockk['adx'] = cut_to1(0.015*talib.ADX(stockk['high'],stockk['low'],stockk['close'], timeperiod=14))
    stockk['adxr'] = cut_to1(0.015*talib.ADXR(stockk['high'],stockk['low'],stockk['close'],timeperiod=14))
    stockk['apo'] = cut_to1(0.5*talib.APO(stockk['close'], fastperiod=12, slowperiod=26, matype=0))
    stockk['ardown'], stockk['arup']= talib.AROON(stockk['high'],stockk['low'],timeperiod=14)
    stockk['ardown']=0.02*stockk['ardown']-1
    stockk['arup']=0.02*stockk['arup']-1
    stockk['arosc'] = 0.01*talib.AROONOSC(stockk['high'],stockk['low'],timeperiod=14)
    stockk['bop'] = talib.BOP(stockk['open'],stockk['high'],stockk['low'],stockk['close'])
    stockk['cci'] = 0.004*talib.CCI(stockk['high'],stockk['low'],stockk['close'],timeperiod=14)
    stockk['cci_d']=0.637*np.arctan(1*stockk['cci'].diff()/(0.1+stockk['cci'].shift()))

    return stockk


def getStockCharacter(stock_data):
    # stockk=getK(stock_data)
    stockk=trans_features(stock_data)

    stockk=stockk.drop(['max20','max60','max120','min20','min60',
        'min120','sma5','sma10','sma20','sma40','sma80',
        "volume",'preclose','amount','avg_price','day_price'],axis=1) 
                        #,'HO','OL','uppwer','downpwer'

    return stockk