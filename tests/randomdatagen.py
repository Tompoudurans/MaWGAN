# -*- coding: utf-8 -*-
import torch
from collections import namedtuple
from random import uniform, randint
import sqlalchemy as sa
import pandas as pd
from makeholes import mkhol

def make_data(number,alt):
    gendata = []
    for i in range(number):
        a = randint(200, 800)/100
        b = randint(0, 10)/1
        c = randint(50, 200)/10
        gendata.append([a,b,c])
    dataset = torch.tensor(gendata)
    if alt:
        hole = mkhol(0.2,gendata)
        datahole = torch.tensor(hole.to_numpy())
    else:
        datahole = None
    return datahole, dataset

def generate_random_testing_data(size):
    """
    create random testing data 3 numrical 2 catgorical and 1 time, with size*3 number of record
    """
    engine = sa.create_engine("sqlite:///flight.db")
    connection = engine.connect()
    Reading = namedtuple("Reading", "flight, temp, pressure, humidity, wind")

    sql = """
    CREATE TABLE readings (
        flight    VARCHAR(10) NOT NULL,
        temp      NUMERIC(3,1) NOT NULL,
        pressure  NUMERIC(4,0) NOT NULL,
        humidity  NUMERIC(3,0) NOT NULL,
        wind    VARCHAR(10) NOT NULL,

        CONSTRAINT temp_ck CHECK (temp BETWEEN -70 AND 70),
        CONSTRAINT pres_ck CHECK (pressure BETWEEN 0 AND 2000),
        CONSTRAINT hum_ck CHECK (humidity BETWEEN 0 AND 100)
    )
    """
    connection.execute(sql)
    winds = ["strong", "medium", "weak"]
    readings = [
        Reading(
            flight=flights,
            temp=round(uniform(23, 27), 1),
            pressure=randint(1020, 1050),
            humidity=randint(30, 50),
            wind=winds[randint(0, 2)],
        )
        for flights in ["hab1", "hab2", "hab3"]
        for i in range(size)
    ]

    sql = """
        INSERT INTO readings
            (flight, temp, pressure, humidity, wind)
        VALUES
            (?, ?, ?, ?, ?)
    """

    for reading in readings:
        values = (
            reading.flight,
            reading.temp,
            reading.pressure,
            reading.humidity,
            reading.wind,
        )
        connection.execute(sql, values)
    return pd.read_sql("readings", connection)


if __name__ == "__main__":
    generate_random_testing_data(10)
