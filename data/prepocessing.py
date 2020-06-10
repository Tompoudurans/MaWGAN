import pandas

def import_penguin(file):
    penguin = pandas.read_csv(file)
    penguin = penguin.replace({'MALE':0.0,'FEMALE':1.0,'.':None})
    penguin = penguin.dropna()
    penguin = penguin.replace({'Chinstrap':0,'Adelie':1,'Gentoo':2})
    penguin = penguin.replace({'Dream':0,'Torgersen':1,'Biscoe':2})
    penguin = penguin.to_numpy('float')
    return penguin
