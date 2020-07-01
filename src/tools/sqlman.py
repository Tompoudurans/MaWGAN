import sqlalchemy as sa
from src.tools.prepocessing import get_norm
import pandas as pd

def load_sql(file,table):
    engine = sa.create_engine('sqlite:///' + file) #creates an empty database
    connection = engine.connect()
    database = pd.read_sql(table, connection)
    # write a line to handel categorical
    database = database.dropna()
    database = database,mean,std = get_norm(database)


def unusable_save_sql():
    readings = [
        Reading(flight='hab1', ts='2015-01-01 09:00:00', temp=24.9, pressure=1020, humidity=40),
        Reading(flight='hab1', ts='2015-01-01 09:01:00', temp=25.1, pressure=1019, humidity=41),
        Reading(flight='hab1', ts='2015-01-01 09:02:00', temp=25.5, pressure=1012, humidity=42),
    ]

    sql = """
    INSERT INTO readings
        (flight, ts, temp, pressure, humidity)
    VALUES
        (?, ?, ?, ?, ?)
    """
    #We can now loop over our readings list and execute our SQL statement once for each entry.
    for reading in readings:
        values = (reading.flight, reading.ts, reading.temp, reading.pressure, reading.humidity)
        connection.execute(sql, values)
    pd.read_sql('readings', connection)
