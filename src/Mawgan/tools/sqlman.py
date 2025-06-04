import sqlalchemy as sa
import pandas as pd

import numpy as np
from scipy.linalg import sqrtm
#___#---------------------------------------------------------------------------
#___#
#___#**Project:**         Masked Wasterstin generative adviersal network\
#___#**Script:**          sqlman.py\
#___#**Author:**          Thomas Poudevigne\
#___#**Date Created:**    21 July 2021\
#___#**Reviewer:**        TBC\
#___#**Devops Feature:**  #[don't know]\
#___#**Devops Backlog:**  #[don't know]\
#___#**Devops Task:**     #[don't know]\
#___#**Devops Repo:**     ganrunner\tools\
#___#**MARS:**            "[don't know]"
#___#
#___#
#____#Description
#____#This srcipt is to load and save sql files
#____#
#___#---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#_#
#__#1. Load sql files
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#this function loads an SQL table from a db file
#_#Reviewer Notes\

def load_sql(file, table):
    """
    Loads an SQL table
    """
    #_#steps\
    #_# acesss the file
    engine = sa.create_engine("sqlite:///" + file)
    #_# read the file
    connection = engine.connect()
    #_# load the table
    database = pd.read_sql(table, connection)
    #_# output the data from the table
    return database


#-------------------------------------------------------------------------------
#_#
#__#2. Load sql files
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#this function saves an SQL table to a db file
#_#Reviewer Notes\

def save_sql(df, file, exists="append"):
    """
    Saves the generated data to a SQL table called generated_data
    """
    #_#steps\
    #_# acesss the file
    engine = sa.create_engine("sqlite:///" + file)
    #_# save the table to that file
    df.to_sql("torch_generated_data", con=engine, if_exists=exists)  # , index=False)


#-------------------------------------------------------------------------------
#_#
#__#3. see table names
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#this function Reads and output all tables names in a sql file
#_#Reviewer Notes\
def all_tables(file):
    """
    Reads all tables in a sql file
    """
    #_#steps\
    #_# acesss the file
    engine = sa.create_engine("sqlite:///" + file)
    #_# read the file
    connection = engine.connect()
    #_# scan the tables
    inspector = sa.inspect(engine)
    #_# outputs table names
    return inspector.get_table_names()
