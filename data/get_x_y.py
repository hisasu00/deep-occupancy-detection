from unicodedata import name
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime as dt
import datetime
import calendar


def get_month_energy(id, year, month):
    dic={}
    num_days  = calendar.monthrange(year, month)[1]

    for day in range(1, num_days+1, 1):
    
        try:
            date_string = datetime.date(year, month, day)
            energy = pd.read_csv('./ecodataset/Energy/0'+str(id)+'/'+str(date_string)+".csv", header=None)
            dic[str(date_string)]=energy[0].values

        except FileNotFoundError:
            print(f"{date_string}:FileNotFoundError")
            print("skip this date")
            pass

    return pd.DataFrame(dic)


def interpolate_missing(df):
    df = df.replace(-1, np.nan)
    for col in df.columns:
        if any(pd.isnull(df[col].values)):
            df[col] = df[col].interpolate(limit=None, limit_direction='both').values
    return df


def to_interval_energy(df, interval):
    dic={}
    for col, vals in df.iteritems():
        intervaled_energy = []
        for t in range(0, 86400, interval):
            intervaled_energy.append(sum(vals.values[t:t+interval]))
        dic[col] = intervaled_energy
    return pd.DataFrame(dic)


def to_interval_occupancy(df, interval):
    dic={}
    for row, vals in df.iterrows():
        intervaled_occupancy=[]
        for t in range(0,86400,interval):
            if np.mean(vals.values[t:t+interval]==1) > 0.80:
                intervaled_occupancy.append(1)
            else:
                intervaled_occupancy.append(0)
        dic[row]=intervaled_occupancy
    return pd.DataFrame(dic)


def get_targret_energy(house_id, target_months, interval_s):
    df_energy = pd.DataFrame()

    for month in target_months:
        year = 2013 if month == 1 else 2012
        
        tmp_energy = get_month_energy(house_id, year, month)
        tmp_energy = interpolate_missing(tmp_energy)
        tmp_energy = to_interval_energy(tmp_energy, interval_s)
        
        df_energy = pd.concat([df_energy, tmp_energy], axis=1)
    return df_energy


def get_numerator(date, energy):
    numerator = energy[date].values
    return numerator


def get_denominator(date, interval, energy):

    # maximum for 2 weeks before and after
    datetime_date = dt.strptime(date, '%Y-%m-%d')
    timedelta_14days = datetime.timedelta(days=14)
    timedelta_1day = datetime.timedelta(days=1)

    denominator = energy.T[str(datetime_date - timedelta_14days) : str(datetime_date + timedelta_14days)].max()
    denominator_1day_before = energy.T[str(datetime_date - timedelta_1day - timedelta_14days) : str(datetime_date - timedelta_1day + timedelta_14days)].max()
    denominator_1day_after = energy.T[str(datetime_date + timedelta_1day - timedelta_14days) : str(datetime_date + timedelta_1day + timedelta_14days)].max()

    # maximum for 30 minutes before and after
    list_1=[]
    T = interval -1 
    T_minus1 = T - 1 
    for time in range(interval):
        if time == 0:
            max_val = max([denominator_1day_before[T], denominator[0], denominator[1]])
            list_1.append(max_val)

        elif time == T:
            max_val  = max([denominator[T_minus1], denominator[T], denominator_1day_after[0]])
            list_1.append(max_val)

        else:
            max_val = max([denominator[time - 1], denominator[time], denominator[time + 1]])
            list_1.append(max_val)
    denominator = list_1

    return denominator


def build_ratio(date_columns, energy, interval):
    dic={}
    for date in date_columns:
        # format date
        date = str(dt.strptime(date, '%d-%b-%Y'))
        where_day_in_string = 10
        date = date[:where_day_in_string]

        # build energy data ratio
        numerator =  get_numerator(date, energy)
        denominator = get_denominator(date, interval, energy)
        ratio = (numerator / denominator).round(7)
        dic[date]=ratio
    return pd.DataFrame(dic)


def get_corresponding_energy(occupancy_columns, energy_df):
    list_1=[]
    where_day_in_string = 10

    for i in occupancy_columns:
        date = str(dt.strptime(i, '%d-%b-%Y'))
        date = date[:where_day_in_string]

        for val in energy_df[date].values:
            list_1.append(val)     
    return list_1


def get_weekdays(target_days):
    target_days= pd.DataFrame(target_days)
    target_days = pd.to_datetime(target_days[0], format="%d-%b-%Y")
    weekdays = target_days.dt.weekday
    weekdays = (weekdays == 6).values
    weekdays = weekdays.astype(np.int)
    weekdays = np.array([[i]*24 for i in weekdays]).reshape(-1)
    return weekdays


def get_am_pm_columns(df):
    
    am_pm = []
    for time in df["Time"]:
        morning = [6, 7, 8, 9, 10]
        lunch = [11, 12, 13, 14, 15, 16]
        if time in morning:
            am_pm.append(0)
        elif time in lunch:
            am_pm.append(1)
        else:
            am_pm.append(2)
    df["Time"] = am_pm
    am_pm = pd.get_dummies(df["Time"])
    df = df.drop(columns="Time")
    df = pd.concat([df, am_pm], axis=1)
    df = df.rename({0: "Am", 1: "Lunch", 2: "Pm"}, axis=1)
    
    return df


def create_features(energy, col):
    mean_list=[]
    max_list=[]
    min_list=[]
    std_list=[]
    range_list=[]
    time_list=[]

    for t in range(0, len(energy), 2):
        mean_list.append(np.mean(energy[t:t+2]))
        max_list.append(np.max(energy[t:t+2]))
        min_list.append(np.min(energy[t:t+2]))
        std_list.append(np.std(energy[t:t+2]))
        range_list.append(abs(energy[t+1] - energy[t]))

    return mean_list, max_list, min_list, std_list, range_list
