from .prepocessing import get_norm
import pandas as pd
from .categorical import encoding


def procsses_sql(database):
    """
    pre-procsses the table, ready to be trained
    """
    database, details = encoding(database)
    col = database.columns
    database, mean, std = get_norm(database)
    return database, [mean, std, details, col]
