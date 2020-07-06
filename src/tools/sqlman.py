import sqlalchemy as sa
from src.tools.prepocessing import get_norm
import pandas as pd


def load_sql(file, table):
    """
    Loads an SQL table and pre-procsses the table, ready to be trained
    """
    engine = sa.create_engine("sqlite:///" + file ".db")
    connection = engine.connect()
    database = pd.read_sql(table, connection)
    database, idexes = factorizing(database)
    database = database.dropna()
    database, mean, std = get_norm(database)
    return database, mean, std ,idexes


def save_sql(df,file):
    """
    Saves the generated data to a SQL table called generated_data
    """
    engine = sa.create_engine("sqlite:///" + file + ".db")
    df = df.drop(columns=["dataset"])
    df.to_sql("generated_data", con=engine, if_exists="append")
    #engine.execute("SELECT * FROM generated_data"").fetchall()


def factorizing(data):
    """
    trasform categorical data into numrical, saves maping on a list.
    """
    indexs = []
    for name in data.columns:
        if "O" == data[name].dtype:
            new = pd.factorize(data[name])
            data[name] = new[0]
            indexs.append(new[1])
    return data, indexs
