import sqlalchemy as sa
import pandas as pd

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


def save_sql(df, file, exists="append"):
    """
    Saves the generated data to a SQL table called generated_data
    """
    engine = sa.create_engine("sqlite:///" + file)
    try:
        df = df.drop(columns=["dataset"])
    except KeyError:
        pass
    df.to_sql("torch_generated_data", con=engine, if_exists=exists)  # , index=False)


def all_tables(file):
    """
    Reads all tables in a sql file
    """
    engine = sa.create_engine("sqlite:///" + file)
    connection = engine.connect()
    inspector = sa.inspect(engine)
    return inspector.get_table_names()
