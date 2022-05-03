import calendar
import datetime
from unicodedata import name

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_month_energy(house_id, year, month):
    month_energy = {}
    num_days  = calendar.monthrange(year, month)[1]
    for day in range(1, num_days+1, 1):
        try:
            date_string = datetime.date(year, month, day)
            energy = pd.read_csv('./ecodataset/Energy/0'+str(house_id)+'/'+str(date_string)+".csv", header=None)
            month_energy[str(date_string)] = energy[0].values
            # energy shape of (60*60*24=86400, 16), the first column is the main trunk value.
        except FileNotFoundError:
            print(f"{date_string}:FileNotFoundError")
            print("skip this date")
            pass
    return pd.DataFrame(month_energy)


def interpolate_missing(energy):
    energy = energy.replace(-1, np.nan)
    # -1 is missing value defined by ECO data set
    for day in energy.columns:
        if any(pd.isnull(energy[day].values)):
            energy[day] = energy[day].interpolate(limit=None, limit_direction='both').values
    return energy


def to_intervals_energy(energy, intervals):
    intervaled_energies = {}
    for day, vals in energy.iteritems():
        intervaled_energy = []
        for t in range(0, 86400, intervals):
            sumed_value = sum(vals.values[t:t+intervals])
            intervaled_energy.append(sumed_value)
        intervaled_energies[day] = intervaled_energy
    return pd.DataFrame(intervaled_energies)


def to_intervals_occupancy(occupancy, intervals):
    intervaled_occupancys = {}
    for day, vals in occupancy.iterrows():
        intervaled_occupancy = []
        for t in range(0, 86400, intervals):
            if np.mean(vals.values[t:t+intervals]==1) > 0.80:
                intervaled_occupancy.append(1)
            else:
                intervaled_occupancy.append(0)
        intervaled_occupancys[day] = intervaled_occupancy
    return pd.DataFrame(intervaled_occupancys)


def get_targret_energy(house_id, target_months, interval_s):
    energies = pd.DataFrame()
    for month in target_months:
        year = 2013 if month == 1 else 2012
        energy = get_month_energy(house_id, year, month)
        energy = interpolate_missing(energy)
        energy = to_intervals_energy(energy, interval_s)
        energies = pd.concat([energies, energy], axis=1)
    return energies


def get_numerator(date, energy):
    numerator = energy[date].values
    return numerator


def get_denominator(date, intervals, energy):

    # maximum for 2 weeks before and after
    datetime_date = datetime.datetime.strptime(date, '%Y-%m-%d')
    timedelta_14days = datetime.timedelta(days=14)
    timedelta_1day = datetime.timedelta(days=1)
    candidate = energy.T[str(datetime_date - timedelta_14days) : str(datetime_date + timedelta_14days)].max()
    # energy.T shape of (num_days, 24)
    candidate_1day_before = energy.T[str(datetime_date - timedelta_1day - timedelta_14days) : str(datetime_date - timedelta_1day + timedelta_14days)].max()
    candidate_1day_after = energy.T[str(datetime_date + timedelta_1day - timedelta_14days) : str(datetime_date + timedelta_1day + timedelta_14days)].max()

    # maximum for 30 minutes before and after
    denominator = []
    T = intervals -1 
    T_minus1 = T - 1 
    for time in range(intervals):
        if time == 0:
            max_val = max([candidate_1day_before[T], candidate[0], candidate[1]])
            denominator.append(max_val)
        elif time == T:
            max_val  = max([candidate[T_minus1], candidate[T], candidate_1day_after[0]])
            denominator.append(max_val)
        else:
            max_val = max([candidate[time - 1], candidate[time], candidate[time + 1]])
            denominator.append(max_val)

    return denominator


def build_ratio(date_columns, energy, intervals):
    ratios={}
    for date in date_columns:
        # format date
        date = str(datetime.datetime.strptime(date, '%d-%b-%Y'))
        where_day_in_string = 10
        date = date[:where_day_in_string]

        # build energy data ratio
        numerator =  get_numerator(date, energy)
        denominator = get_denominator(date, intervals, energy)
        ratio = (numerator / denominator).round(7)
        ratios[date] = ratio
    return pd.DataFrame(ratios)


def get_corresponding_energy(occupancy_columns, energy_df):
    """

    Extract data on days when home conditions are observed,
    from energy data for a certain periods.

    """
    energies = []
    where_day_in_string = 10
    for date in occupancy_columns:
        date = str(datetime.datetime.strptime(date, '%d-%b-%Y'))
        date = date[:where_day_in_string]
        energies += energy_df[date].values.tolist()
    return energies


def get_weekdays(target_days):
    target_days = pd.DataFrame(target_days)
    # target_days shape of (num_days, 1)
    target_days = pd.to_datetime(target_days[0], format="%d-%b-%Y")
    weekdays = target_days.dt.weekday
    weekdays = (weekdays == 6).values
    weekdays = weekdays.astype(np.int)
    weekdays = np.array([[i]*24 for i in weekdays]).reshape(-1)
    return weekdays


def get_am_pm(features):    
    am_pm = []
    morning = [6, 7, 8, 9, 10]
    lunch = [11, 12, 13, 14, 15, 16]
    for time in features["Time"]:
        if time in morning:
            am_pm.append(0)
        elif time in lunch:
            am_pm.append(1)
        else:
            am_pm.append(2)
    features["Time"] = am_pm
    am_pm = pd.get_dummies(features["Time"])
    features = features.drop(columns="Time")
    features = pd.concat([features, am_pm], axis=1)
    features = features.rename({0: "Am", 1: "Lunch", 2: "Pm"}, axis=1)
    return features


def create_features(energy, col):
    means = []
    maxs = []
    mins = []
    stds = []
    ranges = []
    for t in range(0, len(energy), 2):
        means.append(np.mean(energy[t:t+2]))
        maxs.append(np.max(energy[t:t+2]))
        mins.append(np.min(energy[t:t+2]))
        stds.append(np.std(energy[t:t+2]))
        ranges.append(abs(energy[t+1] - energy[t]))
    return means, maxs, mins, stds, ranges
