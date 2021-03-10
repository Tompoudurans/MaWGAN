# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:17:03 2020

@author: drtro
"""
import sqlalchemy as sa
import pandas as pd
import sklearn.metrics


def load_sql(file, table):
    """
    Loads an SQL table
    """
    engine = sa.create_engine("sqlite:///" + file)
    connection = engine.connect()
    database = pd.read_sql(table, connection)
    try:
        return database.drop(columns="index")
    except KeyError:
        return database


r = load_sql("penguin.db", "fullpenguin")
f = load_sql("penguin.db", "torch_generated_data")
col = f.columns
nf = f.iloc[:333]
for c in col:
    print(sklearn.metrics.mutual_info_score(nf[c], r[c]))
rr = round(r)
rf = round(nf)
print("--------------------------")
rr[col[2]] = round(rr[col[2]], -1)
rf[col[2]] = round(rf[col[2]], -1)
rf[col[3]] = round(rf[col[3]], -2)
rr[col[3]] = round(rr[col[3]], -2)
for c in col:
    print(sklearn.metrics.mutual_info_score(rf[c], rr[c]))
