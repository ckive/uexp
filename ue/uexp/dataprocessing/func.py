import datetime
import os
import numpy as np
import urllib, zipfile
from pathlib import Path
from datetime import *
from typing import List
from sklearn.model_selection import train_test_split

TIME_ZONE_SHANGHAI = 'Asia/Shanghai'  ## Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = 'US/Eastern'  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = 'Europe/Paris'  # CAC,
TIME_ZONE_BERLIN = 'Europe/Berlin'  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = 'Asia/Jakarta'  # LQ45
TIME_ZONE_SELFDEFINED = 'xxx'  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)
BINANCE_BASE_URL = 'https://data.binance.vision/'


def calc_time_zone(ticker_list: List[str], time_zone_selfdefined: str,
                   use_time_zone_selfdefined: int) -> str:
    time_zone = ''
    if use_time_zone_selfdefined == 1:
        time_zone = time_zone_selfdefined
    elif ticker_list in [HSI_50_TICKER, SSE_50_TICKER, CSI_300_TICKER]:
        time_zone = TIME_ZONE_SHANGHAI
    elif ticker_list in [DOW_30_TICKER, NAS_100_TICKER, SP_500_TICKER]:
        time_zone = TIME_ZONE_USEASTERN
    elif ticker_list == CAC_40_TICKER:
        time_zone = TIME_ZONE_PARIS
    elif ticker_list in [
            DAX_30_TICKER, TECDAX_TICKER, MDAX_50_TICKER, SDAX_50_TICKER
    ]:
        time_zone = TIME_ZONE_BERLIN
    elif ticker_list == LQ45_TICKER:
        time_zone = TIME_ZONE_JAKARTA
    else:
        raise ValueError("Time zone is wrong.")
    return time_zone


# e.g., '20210911' -> '2021-09-11'
def add_hyphen_for_date(d: str) -> str:
    res = d[:4] + '-' + d[4:6] + '-' + d[6:]
    return res


# e.g., '2021-09-11' -> '20210911'
def remove_hyphen_for_date(d: str) -> str:
    res = d[:4] + d[5:7] + '-' + d[8:]
    return res


# filename: str
# output: stockname
def calc_stockname_from_filename(filename):
    return filename.split("/")[-1].split(".csv")[0]


def calc_all_filenames(path):
    dir_list = os.listdir(path)
    dir_list.sort()
    paths2 = []
    for dir in dir_list:
        filename = os.path.join(os.path.abspath(path), dir)
        if ".csv" in filename and "#" not in filename and "~" not in filename:
            paths2.append(filename)
    return paths2


def calc_stocknames(path):
    filenames = calc_all_filenames(path)
    res = []
    for filename in filenames:
        stockname = calc_stockname_from_filename(filename)
        res.append(stockname)
    return res


def remove_all_files(remove, path_of_data):
    assert remove in [0, 1]
    if remove == 1:
        os.system("rm -f " + path_of_data + "/*")
    dir_list = os.listdir(path_of_data)
    for file in dir_list:
        if "~" in file:
            os.system("rm -f " + path_of_data + "/" + file)
    dir_list = os.listdir(path_of_data)

    if remove == 1:
        if len(dir_list) == 0:
            print("dir_list: {}. Right.".format(dir_list))
        else:
            print("dir_list: {}. Wrong. You should remove all files by hands.".
                  format(dir_list))
        assert len(dir_list) == 0
    else:
        if len(dir_list) == 0:
            print("dir_list: {}. Wrong. There is not data.".format(dir_list))
        else:
            print("dir_list: {}. Right.".format(dir_list))
        assert len(dir_list) > 0


def date2str(dat):
    return datetime.date.strftime(dat, "%Y-%m-%d")


def str2date(str_dat):
    return datetime.datetime.strptime(str_dat, "%Y-%m-%d").date()


### ticker download helpers


def get_destination_dir(file_url):
    store_directory = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(store_directory, file_url)


def get_download_url(file_url):
    return "{}{}".format(BINANCE_BASE_URL, file_url)


#downloads zip, unzips zip and deltes zip
def download_n_unzip_file(base_path, file_name, date_range=None):
    download_path = "{}{}".format(base_path, file_name)
    if date_range:
        date_range = date_range.replace(" ", "_")
        base_path = os.path.join(base_path, date_range)

    #raw_cache_dir = get_destination_dir("./cache/tick_raw")
    raw_cache_dir = "./cache/tick_raw"
    zip_save_path = os.path.join(raw_cache_dir, file_name)

    csv_name = os.path.splitext(file_name)[0] + ".csv"
    csv_save_path = os.path.join(raw_cache_dir, csv_name)

    fhandles = []

    if os.path.exists(csv_save_path):
        print("\nfile already exists! {}".format(csv_save_path))
        return [csv_save_path]

    # make the "cache" directory (only)
    if not os.path.exists(raw_cache_dir):
        Path(raw_cache_dir).mkdir(parents=True, exist_ok=True)

    try:
        download_url = get_download_url(download_path)
        dl_file = urllib.request.urlopen(download_url)
        length = dl_file.getheader('content-length')
        if length:
            length = int(length)
            blocksize = max(4096, length // 100)

        with open(zip_save_path, 'wb') as out_file:
            dl_progress = 0
            print("\nFile Download: {}".format(zip_save_path))
            while True:
                buf = dl_file.read(blocksize)
                if not buf:
                    break
                out_file.write(buf)
                #visuals
                #dl_progress += len(buf)
                #done = int(50 * dl_progress / length)
                #sys.stdout.write("\r[%s%s]" % ('#' * done, '.' * (50-done)) )
                #sys.stdout.flush()

        #unzip and delete zip
        file = zipfile.ZipFile(zip_save_path)
        with zipfile.ZipFile(zip_save_path) as zip:
            #guaranteed just 1 csv
            csvpath = zip.extract(zip.namelist()[0], raw_cache_dir)
            fhandles.append(csvpath)
        os.remove(zip_save_path)
        return fhandles

    except urllib.error.HTTPError:
        print("\nFile not found: {}".format(download_url))
        pass


def convert_to_date_object(d):
    year, month, day = [int(x) for x in d.split('-')]
    date_obj = date(year, month, day)
    return date_obj


def get_path(trading_type,
             market_data_type,
             time_period,
             symbol,
             interval=None):
    trading_type_path = 'data/spot'
    #currently just supporting spot
    if trading_type != 'spot':
        trading_type_path = f'data/futures/{trading_type}'
    if interval is not None:
        path = f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/{interval}/'
    else:
        path = f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/'
    return path


### FROM KAGGLE
# reduce memory (@mfjwr1); distorts the data a little (but reduces by 60% memory)
def red_mem(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) /
                                        start_mem))

    return df

    # Split for TimeSeries


# def TimeSeries_Split(ldf,split_id=[None,None],test_id=False,cut_id=None):

#     # Reduce the number of used data
#     if(cut_id is not None):
#         print('data reduction used')
#         ldf = ldf.iloc[-cut_id:]
#         t1 = ldf.index.max();t0 = ldf.index.min()
#         print(f'Dataset Min.Index: {t0} | Max.Index: {t1}')

#     if(split_id[0] is not None):
#         # General Percentage Split (Non Shuffle requied for Time Series)
#         train_df,pred_df = train_test_split(ldf,test_size=split_id[0],shuffle=False)
#     elif(split_id[1] is not None):
#         # specific time split (NOT USED)
#         train_df = df.loc[:split_id[1]]; pred_df = df.loc[split_id[1]:]
#     else:
#         print('Choose One Splitting Method Only')

# #     y_train = train_df[feature]
# #     X_train = train_df.loc[:, train_df.columns != feature]
# #     if(test_id):
# #         y_test = pred_df[feature]
# #         X_test = pred_df.loc[:, pred_df.columns != feature]

#     return train_df,pred_df # return


def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])


def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']


def gen_feats0(df, row=False):
    df_feat = df
    df_feat['spread'] = df_feat['High'] - df_feat['Low']
    #df_feat['log_price_change'] = np.log(df_feat['Close']/df_feat['Open'])
    df_feat['upper_Shadow'] = upper_shadow(df_feat)
    df_feat['lower_Shadow'] = lower_shadow(df_feat)
    df_feat['trade'] = df_feat['Close'] - df_feat['Open']
    #df_feat['LOGVOL'] = np.log(1. + df_feat['Volume'])
    #df_feat['LOGVOL'] = df_feat['LOGVOL']

    return df_feat


def gen_feats1(df, row=False):
    df_feat = df
    df_feat['spread'] = df_feat['High'] - df_feat['Low']
    df_feat['log_price_change'] = np.log(df_feat['Close'] / df_feat['Open'])
    df_feat['upper_Shadow'] = upper_shadow(df_feat)
    df_feat['lower_Shadow'] = lower_shadow(df_feat)
    df_feat["high_div_low"] = df_feat["High"] / df_feat["Low"]
    df_feat['trade'] = df_feat['Close'] - df_feat['Open']
    df_feat['shadow1'] = df_feat['trade'] / df_feat['Volume']
    df_feat['shadow3'] = df_feat['upper_Shadow'] / df_feat['Volume']
    df_feat['shadow5'] = df_feat['lower_Shadow'] / df_feat['Volume']
    df_feat['mean1'] = (df_feat['shadow5'] + df_feat['shadow3']) / 2
    df_feat['mean2'] = (df_feat['shadow1'] + df_feat['Volume']) / 2
    df_feat['UPS'] = (df_feat['High'] -
                      np.maximum(df_feat['Close'], df_feat['Open']))
    df_feat['UPS'] = df_feat['UPS']
    df_feat['LOS'] = (np.minimum(df_feat['Close'], df_feat['Open']) -
                      df_feat['Low'])
    df_feat['LOS'] = df_feat['LOS']
    df_feat['LOGVOL'] = np.log(1. + df_feat['Volume'])
    df_feat['LOGVOL'] = df_feat['LOGVOL']
    df_feat["Close/Open"] = df_feat["Close"] / df_feat["Open"]
    df_feat["Close-Open"] = df_feat["Close"] - df_feat["Open"]
    df_feat["High-Low"] = df_feat["High"] - df_feat["Low"]
    df_feat["High/Low"] = df_feat["High"] / df_feat["Low"]
    if row: df_feat['Mean'] = df_feat[['Open', 'High', 'Low', 'Close']].mean()
    else:
        df_feat['Mean'] = df_feat[['Open', 'High', 'Low',
                                   'Close']].mean(axis=1)
    df_feat["High/Mean"] = df_feat["High"] / df_feat["Mean"]
    df_feat["Low/Mean"] = df_feat["Low"] / df_feat["Mean"]
    mean_price = df_feat[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    median_price = df_feat[['Open', 'High', 'Low', 'Close']].median(axis=1)
    df_feat['high2mean'] = df_feat['High'] / mean_price
    df_feat['low2mean'] = df_feat['Low'] / mean_price
    df_feat['high2median'] = df_feat['High'] / median_price
    df_feat['low2median'] = df_feat['Low'] / median_price
    return df_feat