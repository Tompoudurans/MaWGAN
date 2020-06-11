import pandas

def import_penguin(file,use_categorical):
    penguin = pandas.read_csv(file)
    penguin = penguin.dropna()
    if use_categorical:
        penguin = penguin.replace({'MALE':0,'FEMALE':1})
        penguin = penguin.replace({'Chinstrap':0,'Adelie':1,'Gentoo':2})
        penguin = penguin.replace({'Dream':0,'Torgersen':1,'Biscoe':2})
    else:
        penguin = penguin.drop(columns=['sex','species','island'])
    penguin = penguin.to_numpy('float')
    return penguin
