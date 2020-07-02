import sqlalchemy as sa
from src.tools.prepocessing import get_norm
import pandas as pd

def load_sql(file,table):
    """
    Loads an SQL table and pre-procsses the table, ready to be trained
    """
    engine = sa.create_engine('sqlite:///' + file) #creates an empty database
    connection = engine.connect()
    database = pd.read_sql(table, connection)
    # write a line to handel categorical
    database = database.dropna()
    database = database,mean,std = get_norm(database)


def save_sql(df):
    """
    Saves the generated data to a SQL table - although now it doesnt seem to work
    """
    engine = sa.create_engine('sqlite:///',echo=False)
    df=df.drop(columns=['dataset'])
    df.to_sql('users', con=engine, if_exists='append')
    engine.execute("SELECT * FROM users").fetchall()
