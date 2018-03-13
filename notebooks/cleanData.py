import os
import warnings
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import scipy
from scipy.special import factorial
import math
import time
from identifySuburbs import TaxiZone, Point, process_zones, haversine_distance, manhattan_distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def getTripData(fileName):
    """Reads trip data from CSV"""
    trip = pd.read_csv(fileName, skipinitialspace=True, dtype={"rate_code": int, "passenger_count": int,"store_and_fwd_flag": str})
    return trip

def getFareData(fileName):
    """Reads fare data from CSV"""
    fare = pd.read_csv(fileName, skipinitialspace=True)
    return fare

def getCleanTripData(trip, filtered=True):
    """Cleans a bit trips data based on coordinates"""
    # We should filter extra long trips here as well.
    if filtered:
        lat_min = 40.5
        lat_max = 41
        long_min = -74.25
        long_max = -73.70
        location_p_long_filter = ((trip.pickup_longitude >= long_min) & (trip.pickup_longitude <= long_max))
        location_p_lat_filter = ((trip.pickup_latitude >= lat_min) & (trip.pickup_latitude <= lat_max))
        location_d_lat_filter = ((trip.dropoff_latitude >= lat_min) & (trip.dropoff_latitude <= lat_max))
        location_d_long_filter = ((trip.dropoff_longitude >= long_min) & (trip.dropoff_longitude <= long_max))
        trip = trip[location_p_long_filter & location_p_lat_filter & location_d_long_filter & location_d_lat_filter]
    return trip

def getCleanFareData(fare, filtered=True):
    """Cleans a bit fare data based on fare amount and payment type"""

    if filtered:
        fare_amount_filter = ((fare.fare_amount >= 2.0) & (fare.fare_amount <= 200.0))
        payment_type_filter = ((fare.payment_type == 'CRD') | (fare.payment_type == 'CSH'))
        fare = fare[fare_amount_filter & payment_type_filter]
    return fare

def getHashedData(fare, trip):
    """Adds new ID field to both databases for quick search requests and merge."""
    start = time.time()
    b = fare['medallion'] + fare['hack_license'] + fare['pickup_datetime']
    c = trip['medallion'] + trip['hack_license'] + trip['pickup_datetime']
    end = time.time()
    fare['id'] = b
    trip['id'] = c
    ids = fare["id"]
    dups = fare[ids.isin(ids[ids.duplicated()])]
    fare = fare[~ids.isin(ids[ids.duplicated()])]
    idst = trip["id"]
    dupst = trip[idst.isin(idst[idst.duplicated()])]
    trip = trip[~idst.isin(idst[idst.duplicated()])]
    print(len(dups.index))
    print(len(dupst.index))
    print("Took time {}".format(end-start))
    return (fare, trip)

def mergeHashedData(trip, fare):
    """Merges two data bases base on ID. Both shold be passed through getHashedData function first."""
    trip = trip.sort('id')
    fare = fare.sort('id')
    result = pd.concat([trip, fare], axis=1)
    return result

def getCleanMergedData(merged_data):
    """For merged data base - cleans based on both coordinates and fare amount."""
    data = merged_data.copy()
    data = getCleanTripData(data)
    data = getCleanFareData(data)
    return data

def getPredictionFeatures(fare):
    """This will include all meaningfull feature which then should be filtered based on prediction.
    """
    alldata = fare.copy()
    alldata['pickup_datetime'] = pd.to_datetime(alldata.pickup_datetime)
    alldata['pickup_day'] = alldata['pickup_datetime'].dt.day
    alldata['pickup_hour'] = alldata['pickup_datetime'].dt.hour
    alldata['pickup_minute'] = alldata['pickup_datetime'].dt.minute
    alldata['pickup_weekday'] = alldata['pickup_datetime'].dt.weekday
    alldata['pickup_weekhour'] = alldata['pickup_datetime'].dt.weekday*24 + alldata['pickup_hour']
    alldata['tip_percent'] = 100*alldata['tip_amount'] / alldata['total_amount']
    alldata = alldata[['id', 'pick_locid', 'drop_locid', 'tip_amount', 'tip_percent',
                'pickup_datetime', 'pickup_day','pickup_hour',
                'pickup_minute','pickup_weekday','pickup_longitude',
                'passenger_count','payment_type', 'fare_amount',
                'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]
    JFK_locid = 132
    LaGuardia_locid = 138
    Newark_locid = 1
    zones = process_zones('./data/taxi_zones/taxi_zones.json')
    airports = [JFK_locid, LaGuardia_locid, Newark_locid]
    manhatten = []
    for z in zones:
        if z.boro == "Manhattan":
            manhatten.append(z.locID)
    manhetten_mapper = {}
    airport_mapper = {}
    for index, row in alldata.iterrows():
        ids = row['id']
        ploc = row['pick_locid']
        dloc = row['drop_locid']
        if ploc in manhatten or dloc in manhatten:
            manhetten_mapper.update({ids: 1})
        else:
            manhetten_mapper.update({ids: 0})

        if ploc in airports or dloc in airports:
            airport_mapper.update({ids: 1})
        else:
            airport_mapper.update({ids: 0})
    col1 = 'InManhattan'
    alldata = alldata.assign(**{col1:np.full(len(alldata.index), int(-2))})

    col2 = 'InAirports'
    alldata = alldata.assign(**{col2:np.full(len(alldata.index), int(-2))})

    s = time.time()
    alldata['InManhattan'] = alldata['id'].map(manhetten_mapper)
    alldata['InAirports'] = alldata['id'].map(airport_mapper)
    alldata.loc[:,'hvsine_pick_drop'] = haversine_distance(alldata['pickup_latitude'].values, alldata['pickup_longitude'].values, alldata['dropoff_latitude'].values, alldata['dropoff_longitude'].values)
    alldata.loc[:,'manh_pick_drop'] = manhattan_distance(alldata['pickup_latitude'].values, alldata['pickup_longitude'].values, alldata['dropoff_latitude'].values, alldata['dropoff_longitude'].values)
    alldata.loc[:,'is_card_payment'] = 1*(alldata['payment_type'] == 'CRD')
    mask_p = (alldata.pick_locid == 400)
    mask_d = (alldata.drop_locid == 400)
    alldata.loc[mask_p, 'pick_locid'] = 0
    alldata.loc[mask_d, 'drop_locid'] = 0
    alldata = alldata.drop(['id', 'pickup_datetime', 'payment_type'],axis=1)
    return alldata

def getMonolithTrainingSet(alldata, target_name, target_scaler=100.0):
    """This will make one training set for training."""
    train = alldata.copy()
    train_target = train[target_name]
    # We predicting this var, so we don't want it in training set
    train = train.drop([target_name], axis=1)
    scaler = StandardScaler().fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), index=train.index.values, columns=train.columns.values)
    pca_scaled = PCA()
    pca_scaled.fit(train_scaled)
    train_pca = pd.DataFrame(pca_scaled.transform(train_scaled))
    train_target = train_target / target_scaler
    return (train_pca, train_target)

def getTrainingTestSet(alldata, target_name, target_scaler=100.0):
    """This routine applies StandardScaler normalisation and PCA for better features."""

    train_len = int(len(alldata.index) * 0.6)
    test_len = int(len(alldata.index) * 0.2)
    crossval_len = int(len(alldata.index) * 0.2)

    #print("{}".format(train_len))
    train = alldata.loc[:train_len, :]
    train_target = train[target_name]
    train = train.drop([target_name], axis=1)
    print("Training to predict {} With Predictors: {}".format(target_name, train.columns.values))
    scaler = StandardScaler().fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), index=train.index.values, columns=train.columns.values)
    pca_scaled = PCA()
    pca_scaled.fit(train_scaled)
    train_pca = pd.DataFrame(pca_scaled.transform(train_scaled), index=train.index.values, columns=train.columns.values)

    test = alldata.loc[train_len:train_len+test_len, :]
    test_target = test[target_name]
    test = test.drop([target_name], axis=1)
    test_scaled = pd.DataFrame(scaler.transform(test), index=test.index.values, columns=test.columns.values)
    test_pca = pd.DataFrame(pca_scaled.transform(test_scaled), index=test.index.values, columns=test.columns.values)

    cross_val = alldata.loc[train_len+test_len:, :]
    cross_val_target = cross_val[target_name]
    cross_val = cross_val.drop([target_name], axis=1)
    cross_val_scaled = pd.DataFrame(scaler.transform(cross_val), index=cross_val.index.values, columns=cross_val.columns.values)
    cross_val_pca = pd.DataFrame(pca_scaled.transform(cross_val_scaled), index=cross_val.index.values, columns=cross_val.columns.values)

    train_target = train_target / target_scaler
    test_target = test_target / target_scaler
    cross_val_target = cross_val_target / target_scaler
    return [(train_pca, train_target), (test_pca, test_target), (cross_val_pca, cross_val_target)]

if __name__ == "__main__":
    print("Clean data")
