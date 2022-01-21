from ue.uexp.dataprocessing.processor_binance import BinanceProcessor
import numpy as np
import pandas as pd

from backtesting import Strategy

# sRSI Calculation
# From Method 2
def EMA(data, period=20, column='Close'):
    return data[column].ewm(span=period, adjust=False).mean()

def sRSI(data, period=14, column='Close'):
    delta = data[column].diff(1)
    delta = delta.dropna()
    up = delta.copy()
    down = delta.copy()
    up[up<0]=0
    down[down>0]=0
    data['up'] = up
    data['down'] = down
    AVG_Gain = EMA(data, period, column='up')
    AVG_Loss = abs(EMA(data,period,column='down'))
    RS = AVG_Gain / AVG_Loss
    RSI = 100.0 - (100.0/(1.0 + RS))

    sRSI = (RSI-RSI.rolling(period).min()) / (RSI.rolling(period).max() - RSI.rolling(period).min())
    return sRSI * 100

# hold 1 position at a time, either buy/sell strictly 0.2/0.8

class Simple_sRSI(Strategy):
    d_rsi = 14  # Daily RSI lookback periods
    buy_thresh = 20
    sell_thresh = 80
     
    def init(self):
        # sRSI
        #df['sRSI'] = sRSI(df)
        self.sRSI = self.I(sRSI, self.data.df)
        
    def next(self):
        #https://algotrading101.com/learn/backtrader-for-backtesting/
        if not self.position:
            #look to make a short/long 
            
            # If sRSI < 20, enter long.
            if self.sRSI < self.buy_thresh:

                # 20% sl 沉得住气...okay maybe not 0.8, try 0.9, try 0.92, no stop loss
                #self.buy(sl=.92 * price)
                self.buy()

            # If sRSI > 80, close position
            # close the position, if any.
            elif self.sRSI > self.sell_thresh:
                self.sell()
        else:
            #we have a position
            if self.position.is_long and self.sRSI > self.sell_thresh:
                #close when sRSI > 80  
                self.position.close()
            elif self.sRSI < self.buy_thresh:
                #a short position:
                self.position.close()

class StopLoss_sRSI(Strategy):
    d_rsi = 14  # Daily RSI lookback periods
    buy_thresh = 20
    sell_thresh = 80
    stoploss = 0.92 #optimize this with bt.optimize
     
    def init(self, ):
        # sRSI
        #df['sRSI'] = sRSI(df)
        self.sRSI = self.I(sRSI, self.data.df)
        self._sl = self.stoploss 
        
    def next(self):
        price = self.data.Close[-1]

        if not self.position:
            #look to make a short/long 
            
            # If sRSI < 20, enter long (with stop loss)
            if self.sRSI < self.buy_thresh:
                self.buy(sl=self._sl * price)

            # If sRSI > 80, close position
            # close the position, if any.
            elif self.sRSI > self.sell_thresh:
                self.sell()
        else:
            #we have a position
            if self.position.is_long and self.sRSI > self.sell_thresh:
                #add a decay here
                #close when sRSI > 80 or held for more than 6 hours 
                # (idk how to implement this yet)
                self.position.close()
            elif self.sRSI < self.buy_thresh:
                #a short position:
                self.position.close()


class MultiplePosn_sRSI(Strategy):
    d_rsi = 14  # Daily RSI lookback periods
    buy_thresh = 20
    sell_thresh = 80
    
     
    def init(self, df):
        # sRSI
        #df['sRSI'] = sRSI(df)
        self.sRSI = self.I(sRSI, df)
    
        
    def next(self):
        #https://algotrading101.com/learn/backtrader-for-backtesting/
        if not self.position:
            #look to make a short/long 
            
            # If sRSI < 20, enter long.
            if self.sRSI < self.buy_thresh:

                # 20% sl 沉得住气...okay maybe not 0.8, try 0.9, try 0.92, no stop loss
                #self.buy(sl=.92 * price)
                self.buy()

            # If sRSI > 80, close position
            # close the position, if any.
            elif self.sRSI > self.sell_thresh:
                self.sell()
        else:
            #we have a position
            if self.position.is_long and self.sRSI > self.sell_thresh:
                #add a decay here
                #close when sRSI > 80 or held for more than 6 hours 
                # (idk how to implement this yet)
                self.position.close()
            elif self.sRSI < self.buy_thresh:
                #a short position:
                self.position.close()


class Simple_sRSI_decay(Strategy):
    """
    using Simple_sRSI but with decay factor on thresholds based on hold_time 

    """
    d_rsi = 14  # Daily RSI lookback periods
    buy_thresh = 20
    sell_thresh = 80
    
     
    def init(self):
        # sRSI
        #df['sRSI'] = sRSI(df)
        self.sRSI = self.I(sRSI, df)
    
        
    def next(self):
        #https://algotrading101.com/learn/backtrader-for-backtesting/
        if not self.position:
            #look to make a short/long 
            
            # If sRSI < 20, enter long.
            if self.sRSI < self.buy_thresh:

                # 20% sl 沉得住气...okay maybe not 0.8, try 0.9, try 0.92, no stop loss
                #self.buy(sl=.92 * price)
                self.buy()

            # If sRSI > 80, close position
            # close the position, if any.
            elif self.sRSI > self.sell_thresh:
                self.sell()
        else:
            #we have a position
            if self.position.is_long and self.sRSI > self.sell_thresh:
                #add a decay here
                #close when sRSI > 80 or held for more than 6 hours 
                # (idk how to implement this yet)
                self.position.close()
            elif self.sRSI < self.buy_thresh:
                #a short position:
                self.position.close()