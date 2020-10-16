import sqlalchemy as sa
from .prepocessing import get_norm
import pandas as pd
from .categorical import encoding


def load_sql(file, table):
    """
    Loads an SQL table and pre-procsses the table, ready to be trained
    """
    engine = sa.create_engine("sqlite:///" + file)
    connection = engine.connect()
    database = pd.read_sql(table, connection)
    return database

def procsses_sql(database):
    database, details = encoding(database)
    database = database.dropna()
    col = database.columns
    database, mean, std = get_norm(database)
    return database, mean, std, details, col


def save_sql(df, file, exists="append"):
    """
    Saves the generated data to a SQL table called generated_data
    """
    engine = sa.create_engine("sqlite:///" + file)
    try:
        df = df.drop(columns=["dataset"])
    except KeyError:
        pass
    df.to_sql("generated_data", con=engine, if_exists=exists)
