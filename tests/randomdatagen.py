# -*- coding: utf-8 -*-

from collections import namedtuple
from random import uniform, randint
import sqlalchemy as sa
import pandas as pd


def generate_random_testing_data(size):
    """
    create random testing data 3 numrical
    """
    engine = sa.create_engine("sqlite:///flight.db")
    connection = engine.connect()
    Reading = namedtuple("Reading", "temp, pressure, humidity")

    sql = """
    CREATE TABLE readings (
        temp      NUMERIC(3,1) NOT NULL,
        pressure  NUMERIC(4,0) NOT NULL,
        humidity  NUMERIC(3,0) NOT NULL,

        CONSTRAINT temp_ck CHECK (temp BETWEEN -70 AND 70),
        CONSTRAINT pres_ck CHECK (pressure BETWEEN 0 AND 2000),
        CONSTRAINT hum_ck CHECK (humidity BETWEEN 0 AND 100)
    )
    """
    connection.execute(sql)
    readings = [
        Reading(
            temp=round(uniform(23, 27), 1),
            pressure=randint(1020, 1050),
            humidity=randint(30, 50),
        )
        for i in range(size)
    ]

    sql = """
        INSERT INTO readings
            (temp, pressure, humidity)
        VALUES
            (?, ?, ?)
    """

    for reading in readings:
        values = (reading.temp, reading.pressure, reading.humidity)
        connection.execute(sql, values)
    return pd.read_sql("readings", connection)
