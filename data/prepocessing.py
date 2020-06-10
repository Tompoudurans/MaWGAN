import pandas

def import_penguin(file):
    penguin = pandas.read_csv(file)
    penguin = penguin.dropna()
    penguin = penguin.replace({'MALE':0,'FEMALE':1})
    penguin = penguin.replace({'Chinstrap':0,'Adelie':1,'Gentoo':2})
    penguin = penguin.replace({'Dream':0,'Torgersen':1,'Biscoe':2})
    return penguin