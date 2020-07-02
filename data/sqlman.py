import sqlalchemy as sa
from src.tools.prepocessing import get_norm
import pandas as pd

def load_sql(file,table):
    engine = sa.create_engine('sqlite:///' + file) #creates an empty database
    connection = engine.connect()
    database = pd.read_sql(table, connection)
    # write a line to handel categorical
    database = database.dropna()
    database,mean,std = get_norm(database)
    return database,mean,std


def save_sql(df):
    engine = sa.create_engine('sqlite:///',echo=False)
    df=df.drop(columns=['dataset'])
    df.to_sql('users', con=engine, if_exists='append')
    engine.execute("SELECT * FROM users").fetchall()
