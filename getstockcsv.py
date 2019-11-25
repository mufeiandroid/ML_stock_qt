import newfeature
import numpy as np
import pandas as pd
import tushare as ts
import baostock as bs
import talib
import time
import bcolz

today=time.strftime('%Y-%m-%d',time.localtime(time.time()))

def get_quantiles(data,x): 
    bins=np.r_[-1e100,[np.round(np.quantile(data, i/x),3) for i in range(1,x)],1e100]  # [负无穷，data中result这列的1,1/2,1/3...，正无穷]
    return bins

def precode(c): #在代码后载入sz，sh，6开头代表上证，加sh.前缀
    return 'sh.'+c  if c[0:1]=='6' else 'sz.'+c 


def cut_to1(alpha,minx=-1,maxx=1):
    alpha=alpha.copy()
    alpha[alpha>maxx]=maxx  # 涨幅超过多少，则截取， 应当是在对异常值进行处理
    alpha[alpha<minx]=minx  # 跌幅超过多少，则截取
    return alpha


def getbaostock(symbol, start='2015-01-01', end='2020-10-25', ktype='D',index=0):
    # index=0代表大盘，1代表个股
    infields="date,close,open,preclose,low,high,turn,volume,amount,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST" if index==0 else "date,open,high,low,close,preclose,volume,amount,pctChg"
    rs = bs.query_history_k_data_plus(symbol, infields,start_date=start, end_date=end,
                                    frequency=ktype, adjustflag="2")
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    if len(result)>50:
    
        result=result.sort_values('date').set_index('date').fillna(0)
        result=result[result['volume']!='0']  # 剔除交易量为0数据（即停牌日期数据）
        result=result[result['volume']!='']
        result=result.astype('float32')
        if index==0:  # 拿个股数据，才执行
            result=result[result['psTTM']>0]  # 剔除动态市盈率小于0数据
            
            # 特征工程，通过老特征组合，获得新增特征
            # ？？？？？？？
            result['flowmkt']=0.637*np.arctan(np.log(0.00000001*result['close']*result['volume']/result['turn']))
            result['everyprofit']=0.637*np.arctan(10/result.peTTM)
            result['turn']=0.637*np.arctan((result.turn/10)**0.2)
            result['everypb']=0.637*np.arctan(2/result.pbMRQ)
            result['eversale']=0.637*np.arctan(2/np.sqrt(result.psTTM))
            result['evermoney']=0.637*np.arctan(20/result.pcfNcfTTM)
            result.fillna(0,inplace=True)
            # result.fillna(0,inplace=True)

    # result['p_change']=result['pctChg']
    # print(symbol,result.shape)

        return result 
    else:
        return []

def saveData(path,symbol,data,szzs,days,binscount,r=1,isfeature=1):
     """
     path:文件存储路径
     symbol：股票代码
     data：个股数据
     szzs：上证指数数据
     days：默认为5，代表交易日
     
       
     """
    #如果是用talib、newfeatures计算的数据，需要解除前120条数据
    data.dropna(axis=0,inplace=True)
    # data['amount']=0.25*(data.open+data.close+data.high+data.low)*data['volume']
    data_price=data.values

    # 特征工程，组合生成新特征（有些特征往上证指数szzs数据加，又通过szzs新特征算出个股新特征，有些往个股数据加）
    data['szpctchg1']=18*szzs['close']/szzs['close'].shift(1)-18  # 今日收盘/昨日收盘 - 1  即18*(当日涨幅 - 1)
    data['szpctchg2']=19*szzs['close'].rolling(2).mean()/szzs['close'].shift(2)-19  # 19*(前3日收盘价均值(含当日)/前2日收盘价 - 1)
    data['szpctchg4']=10*szzs['close'].rolling(2).mean()/szzs['close'].shift(4)-10  # 10*(前3日收盘价均值(含当日)/前4日收盘价 - 1)
    data['szpctchg8']=10*szzs['close'].rolling(4).mean()/szzs['close'].shift(8)-10  # 10*(前4日收盘价均值(含当日)/前8日收盘价 - 1)
    data['szpctchg15']=8*szzs['close'].rolling(5).mean()/szzs['close'].shift(15)-8
    data['szpctchg30']=6*szzs['close'].rolling(10).mean()/szzs['close'].shift(30)-6


    szzs['sma5']=szzs['close'].rolling(5).mean() # 大盘5日均线
    szzs['sma10']=szzs['close'].rolling(10).mean()
    szzs['sma20']=szzs['close'].rolling(20).mean()
    szzs['sma40']=szzs['close'].rolling(40).mean()
    szzs['sma80']=szzs['close'].rolling(80).mean()

    szzs['max20']=szzs['high'].rolling(20).max().shift()  # 大盘20日最高价
    szzs['max60']=szzs['high'].rolling(60).max().shift()
    szzs['max120']=szzs['high'].rolling(120).max().shift()

    szzs['min20']=szzs['low'].rolling(20).min().shift()  # 大盘20日最低价
    szzs['min60']=szzs['low'].rolling(60).min().shift()
    szzs['min120']=szzs['low'].rolling(120).min().shift()

    data['zs20_min']=cut_to1(2*szzs['close']/szzs['min20']-2)###  大盘2倍的20日内较最低价的涨幅（涨幅超50截取，或跌幅超50%被截取）
    data['zs60_min']=cut_to1(2*szzs['close']/szzs['min60']-2)###
    data['zs120_min']=cut_to1(2*szzs['close']/szzs['min120']-2)###

    data['zs20_max']=cut_to1(4*szzs['close']/szzs['max20']-4)###  大盘4倍的20日内较最高价的跌幅(涨幅超过25%截取，跌幅超过25%截取)
    data['zs60_max']=cut_to1(4*szzs['close']/szzs['max60']-4)###
    data['zs120_max']=cut_to1(4*szzs['close']/szzs['max120']-4)###

    data['zs5_d']=cut_to1(20*szzs['sma5'].diff()/szzs['sma5'].shift())  # 大盘5日均线差分/昨日5日均线值，涨跌幅限制-5% ~ 5%截取
    data['zs10_d']=cut_to1(40*szzs['sma10'].diff()/szzs['sma10'].shift())
    data['zs20_d']=cut_to1(60*szzs['sma20'].diff()/szzs['sma20'].shift())
    data['zs40_d']=cut_to1(80*szzs['sma40'].diff()/szzs['sma40'].shift())
    data['zs80_d']=cut_to1(100*szzs['sma80'].diff()/szzs['sma80'].shift())

    data['zsr5']=cut_to1(10*szzs['close']/szzs['sma5']-10)  # 大盘收盘价偏离5日均线的幅度，涨跌幅限制-10% ~ 10%截取
    data['zsr10']=cut_to1(8*szzs['close']/szzs['sma10']-8)  
    data['zsr20']=cut_to1(6*szzs['close']/szzs['sma20']-6)
    data['zsr40']=cut_to1(6*szzs['close']/szzs['sma40']-6)
    data['zsr80']=cut_to1(4*szzs['close']/szzs['sma80']-4)
    
    szzs['r5_20']=(4*szzs['sma5']/szzs['sma20']-4)  # 大盘5日均线偏离20日均线幅度，涨跌幅限制-25% ~ 25%截取
    szzs['r10_40']=(4*szzs['sma10']/szzs['sma40']-4)
    szzs['r20_80']=(4*szzs['sma20']/szzs['sma80']-4)
    szzs['r5_40']=(4*szzs['sma5']/szzs['sma40']-4)
    
    data['zsr5_20_d']=cut_to1(10*szzs['r5_20'].diff())  # 大盘5日均线较20日均线的偏移幅度的变化值，反应5日和20日均偏离的剧烈程度
    data['zsr10_40_d']=cut_to1(20*szzs['r10_40'].diff())
    data['zsr20_80_d']=cut_to1(20*szzs['r20_80'].diff())
    data['zsr5_20']=cut_to1(szzs['r5_20'])
    data['zsr10_40']=cut_to1(1*szzs['r10_40'])
    data['zsr20_80']=cut_to1(1*szzs['r20_80'])
    data['zsr5_40']=cut_to1(1*szzs['r5_40'])


    szzs['v5']=(10*szzs['volume']/(0.1+szzs['volume'].rolling(5).mean())) # 大盘交易量较5日均交易量均值偏移幅度，0.1是考虑到rolling前几个值为null
    szzs['v10']=(10*szzs['volume']/(0.1+szzs['volume'].rolling(10).mean()))  # 衡量交易量较平均值异常增减的情况
    szzs['v20']=(10*szzs['volume']/(0.1+szzs['volume'].rolling(20).mean()))
    szzs['v40']=(10*szzs['volume']/(0.1+szzs['volume'].rolling(40).mean()))
    szzs['v80']=(10*szzs['volume']/(0.1+szzs['volume'].rolling(80).mean()))
    
    data['zsv3_9']=(10*szzs['volume'].rolling(3).mean()/(0.1+szzs['volume'].rolling(9).mean()))  # 大盘3日交易量较9日均交易量偏移幅度
    data['zsv5_20']=(10*szzs['volume'].rolling(5).mean()/(0.1+szzs['volume'].rolling(20).mean()))
    data['zsv9_50']=(10*szzs['volume'].rolling(9).mean()/(0.1+szzs['volume'].rolling(50).mean()))
    
    data['zsv5_d']=0.6366*np.arctan(0.2*szzs['v5'].diff())  # 交易量较交易量均线的偏置幅度变化，衡量交易量的变动情况
    data['zsv10_d']=0.6366*np.arctan(0.1*szzs['v10'].diff())  # arctan()用于归一化，把任何取值范围数据压缩到0-1
    data['zsv20_d']=0.6366*np.arctan(0.2*szzs['v20'].diff())  # 为何用0.6366？？？？？？
    data['zsv40_d']=0.6366*np.arctan(0.2*szzs['v40'].diff())
    data['zsv80_d']=0.6366*np.arctan(0.2*szzs['v80'].diff())
    
    data['zsv3_9_d']=0.6366*np.arctan(2*data['zsv3_9'].diff())
    data['zsv5_20_d']=0.6366*np.arctan(2*data['zsv5_20'].diff())
    data['zsv9_50_d']=0.6366*np.arctan(2*data['zsv9_50'].diff())
    
    data['zsv3_9']=0.6366*np.arctan(0.2*data['zsv3_9'])
    data['zsv5_20']=0.6366*np.arctan(0.2*data['zsv5_20'])
    data['zsv9_50']=0.6366*np.arctan(0.2*data['zsv9_50'])
    data['zsv5']=0.6366*np.arctan(0.2*szzs['v5'])
    data['zsv10']=0.6366*np.arctan(0.2*szzs['v10'])
    data['zsv20']=0.6366*np.arctan(0.2*szzs['v20'])
    data['zsv40']=0.6366*np.arctan(0.2*szzs['v40'])
    data['zsv80']=0.6366*np.arctan(0.2*szzs['v80'])

    # 大盘收盘价macd
    szzs['dif'], szzs['dea'],szzs['hist'] = talib.MACD(szzs['close'].astype(float).values, fastperiod=12, slowperiod=26, signalperiod=9)
    data['zsdif']=cut_to1(0.01*szzs['dif'])  # macd的dif线，上下限100截取
    data['zsdea']=cut_to1(0.01*szzs['dea'])

    data['zshist_d']=cut_to1(0.1*szzs['hist'].diff())  # macd红绿柱的差分
    data['zshist']=cut_to1(0.02*szzs['hist'])

    
    # 提取标签值    
    for i in [2,4,7,10]:,
        data['result']=data['close'].shift(-i*r)/data['close']-1  #　ｒ＝１，个股未来2日涨幅(收盘价算)，4日后涨幅，7日后涨幅，10日后涨幅
        # data['result2zs']=data['result']-szzs['close'].shift(-i*r)/szzs['close']
        data['resultmax']=data['close'].rolling(i*r).max().shift(-i*r)/data['close']-1  # 个股未来(2,4,7,10日)最高涨幅(收盘价算)
        data['resultmin']=data['close'].rolling(i*r).min().shift(-i*r)/data['close']-1  # 个股未来(2,4,7,10日)最大跌幅
        data['resultrel']=data['resultmax']+data['resultmin']  # 个股未来(2,4,7,10日)振幅
        remax=data['close'].rolling(i*r).max().shift(-i*r)  # 个股未来(2,4,7,10日)最高价
        remin=data['close'].rolling(i*r).min().shift(-i*r)  # 个股未来(2,4,7,10日)最低价
        reavg=remax-remin  # 个股未来(2,4,7,10日)最高最低价差
        
        if i>2:
            # 每种标签值，对应不同的买入信号
            # 最高最低价均分5份or3份，收盘价低于最低价到1/5的值的权重为4,而1/5到1/3的值是2，中间是0，2/3到5/4的值是-2，4/5到1的值是-4
            buysignal=2*(data['close']<(remin+reavg/5))+2*(data['close']<(remin+reavg/3))-2*(data['close']>(remax-reavg/3))-2*(data['close']>(remax-reavg/5))
            # 进一步约束买入信号，振幅要大于3%。然后如果振幅大于5%，且收盘价还在高低价差下方五分之一内，则buysignal所有权重再加1
            # 最前面加4，因为权重最小值就是4，相当于把权重全部转为正数
            data['resultbuy'+str(i)]=4+(reavg/data['close']>0.03)*(buysignal)+1*(reavg/data['close']>0.05)*(data['close']<(remin+reavg/5))
            # data[(data['resultmax']-data['resultrel']/4)<0].loc[:,'buyorsale'+str(i)]=-1
        # 获取result的1,1/2,1/3...1/9分位值
        bins=get_quantiles(data['result'].dropna(),10)  # data["result"]之前是依据shift获取，因此有null值
        data['resultclass'+str(i)]=pd.cut(data['result'], bins, right=False,labels=False)  # 新建结果类别列，以历史涨幅走势来分类别
        # bins=get_quantiles(data['resultmin'].dropna(),10)
        # data['resultmin'+i]=pd.cut(data['resultmin'], bins, right=False,labels=False)
        # bins=get_quantiles(data['resultmax'].dropna(),10)
        # data['resultmax'+i]=pd.cut(data['resultmax'], bins, right=False,labels=False)
        bins=get_quantiles(data['resultrel'].dropna(),10)
        data['resultrelclass'+str(i)]=pd.cut(data['resultrel'], bins, right=False,labels=False)  # 新建结果类别列，以历振幅走势来分类别

    if isfeature==0: 
    ###这是不用feature的操作
        start=0
        data=data[start:].drop(['ma5','ma10','ma20','result','resultmax','resultmin','resultrel','peTTM','pbMRQ','psTTM',
                                                                  'pcfNcfTTM','p_change','price_change'],axis=1)


###下面是用features的操作
    elif isfeature==1: 
        start=120
        data=newfeature.getStockCharacter(data).astype('float16')
        # data=data[start:].drop(['ma5','ma10','ma20','v_ma5','v_ma10','v_ma20','result','p_change','price_change'],axis=1)
        data=data[start:].drop(['peTTM','pbMRQ','psTTM','pcfNcfTTM','result','resultmax','resultmin','resultrel'],axis=1)
        # data.drop(['pctChg'],axis=1,inplace=True)
    return data

import os, tarfile
#一次性打包整个根目录。空子目录会被打包。
#如果只打包不压缩，将"w:gz"参数改为"w:"或"w"即可。
def make_targz(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
      tar.add(source_dir, arcname=os.path.basename(source_dir))
#逐个添加文件打包，未打包空子目录。可过滤文件。
#如果只打包不压缩，将"w:gz"参数改为"w:"或"w"即可。
def make_targz_one_by_one(output_filename, source_dir):
    tar = tarfile.open(output_filename,"w:gz")
    for root,dir,files in os.walk(source_dir):
      for file in files:
          pathfile = os.path.join(root, file)
          tar.add(pathfile)
    tar.close()


bs.login()



path='/www/stocks/'
path='/Users/hongyuouyang/python/finance/stockdata/'
days=5
r=1
sequence_length=10
stocks=ts.get_stock_basics().sort_index()
stocks.to_csv(path+'stocks.csv')
stocks=stocks.index.values

# szzs=ts.get_hist_data('sh').sort_values('date')
szzs=getbaostock('sh.000001','2016-01-01', today, 'D',1)
j=0
for symbol in stocks[:]:  # symbol代表6位股票代码
    if not(os.path.exists(path+symbol)):

        # data=ts.get_hist_data(symbol)


        #拼合数据生成bcolz文件
        data=getbaostock(precode(symbol),'2016-01-01', today, 'D',0)  # 返回该股票2016年至今的日K数据，df格式，包含十多项指标
        if data is None:
            print(symbol,'no data')
        else:
            if len(data) > 300:
                print(j,symbol,data.shape)
                j += 1
                data=data.sort_values('date')  # 升序排列，日期从2016-现在排序
                datanew=saveData(path,symbol,data,szzs,days,10,r).astype('float16')
                if j==1:
                    datanew.head(1).to_csv(path+'data.csv')

            	datanew.to_csv(path+symbol+'.csv')  # 保存每一只股票的数据到各自的csv文件（包含各种指标及特征）


# make_targz('/www/wwwroot/fina.ouyanghome.com/public/stockdata.tar.gz', path)
# print('data done')
# bs.logout()

#存磁盘