import calendar
import datetime

import numpy as np
import pandas as pd


def get_month_energy(house_id, year, month):
    month_energies = {}
    num_days  = calendar.monthrange(year, month)[1]
    for day in range(1, num_days+1, 1):
        try:
            date_string = datetime.date(year, month, day)
            path = './ecodataset/Energy/0'+str(house_id)+'/'+str(date_string)+".csv"
            energy = pd.read_csv(path, header=None)
            month_energies[str(date_string)] = energy[0].values
            # energy shape of (60*60*24=86400, 16), the first column is the main trunk value.
        except FileNotFoundError:
            print(f"{date_string}:FileNotFoundError")
            print("skip this date")
    return pd.DataFrame(month_energies)


def interpolate_missing(energies):
    energies = energies.replace(-1, np.nan)
    # -1 is missing value defined by ECO data set
    for day in energies.columns:
        if any(pd.isnull(energies[day].values)):
            energies[day] = energies[day].interpolate(limit=None, limit_direction='both').values
    return energies


def to_intervals_energy(energies, intervals):
    intervaled_energies = {}
    for day, vals in energies.iteritems():
        intervaled_energy = []
        for time in range(0, 86400, intervals):
            sumed_value = sum(vals.values[time:time+intervals])
            intervaled_energy.append(sumed_value)
        intervaled_energies[day] = intervaled_energy
    return pd.DataFrame(intervaled_energies)


def to_intervals_occupancy(occupancies, intervals):
    intervaled_occupancys = {}
    for day, vals in occupancies.iterrows():
        intervaled_occupancy = []
        for time in range(0, 86400, intervals):
            if np.mean(vals.values[time:time+intervals]==1) > 0.80:
                intervaled_occupancy.append(1)
            else:
                intervaled_occupancy.append(0)
        intervaled_occupancys[day] = intervaled_occupancy
    return pd.DataFrame(intervaled_occupancys)


def get_target_energy(house_id, target_months, intervals):
    target_energies = pd.DataFrame()
    for month in target_months:
        year = 2013 if month == 1 else 2012
        energies = get_month_energy(house_id, year, month)
        energies = interpolate_missing(energies)
        energies = to_intervals_energy(energies, intervals)
        target_energies = pd.concat([target_energies, energies], axis=1)
    return target_energies


def get_numerator(date, energies):
    numerator = energies[date].values
    return numerator


def get_denominator(date, energies, intervals=24):

    # maximum for 2 weeks before and after
    datetime_date = datetime.datetime.strptime(date, '%Y-%m-%d')
    timedelta_14days = datetime.timedelta(days=14)
    timedelta_1day = datetime.timedelta(days=1)
    candidate = energies.T[str(datetime_date - timedelta_14days):
                         str(datetime_date + timedelta_14days)].max()
    # energy.T shape of (num_days, 24)
    candidate_1day_before = energies.T[str(datetime_date - timedelta_1day - timedelta_14days):
                                     str(datetime_date - timedelta_1day + timedelta_14days)].max()
    candidate_1day_after = energies.T[str(datetime_date + timedelta_1day - timedelta_14days):
                                    str(datetime_date + timedelta_1day + timedelta_14days)].max()
    # maximum for 60(default) minutes before and after
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


def build_ratio(date_columns, energies, intervals=24):
    ratios = {}
    for date in date_columns:
        # format date
        date = str(datetime.datetime.strptime(date, '%d-%b-%Y'))
        where_day_in_string = 10
        date = date[:where_day_in_string]

        # build energy data ratio
        numerator =  get_numerator(date, energies)
        denominator = get_denominator(date, energies, intervals)
        ratio = (numerator / denominator).round(7)
        ratios[date] = ratio
    return pd.DataFrame(ratios)


def get_corresponding_energy(occupancy_columns, energies):
    """

    Extract data on days when home conditions are observed,
    from energy data for a certain periods.

    """
    corresponding_energies = []
    where_day_in_string = 10
    for date in occupancy_columns:
        date = str(datetime.datetime.strptime(date, '%d-%b-%Y'))
        date = date[:where_day_in_string]
        corresponding_energies += energies[date].values.tolist()
    return corresponding_energies


def get_issunday(target_days):
    target_days = pd.DataFrame(target_days)
    # target_days shape of (num_days, 1)
    target_days = pd.to_datetime(target_days[0], format="%d-%b-%Y")
    is_sunday = target_days.dt.weekday
    is_sunday = (is_sunday == 6).values
    is_sunday = is_sunday.astype(np.int)
    is_sunday = np.array([[i]*24 for i in is_sunday]).reshape(-1)
    return is_sunday


def get_am_pm(times):
    am_pm = []
    morning = [6, 7, 8, 9, 10]
    lunch = [11, 12, 13, 14, 15, 16]
    for time in times:
        if time in morning:
            am_pm.append(0)
        elif time in lunch:
            am_pm.append(1)
        else:
            am_pm.append(2)
    am_pm = pd.DataFrame(am_pm)
    am_pm = pd.get_dummies(am_pm[0])
    return am_pm


def create_features(energy):
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
