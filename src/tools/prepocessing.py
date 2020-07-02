import pandas

def import_penguin(file,use_categorical):
    """
    Imports the penguin dataset then processes the data so that it is ready to be trained.
    It sets the categorical data into numerical data
    """
    penguin = pandas.read_csv(file)
    penguin = penguin.dropna()
    if use_categorical:
        penguin = penguin.replace({'MALE':0,'FEMALE':1})
        penguin = penguin.replace({'Chinstrap':0,'Adelie':1,'Gentoo':2})
        penguin = penguin.replace({'Dream':0,'Torgersen':1,'Biscoe':2})
    else:
        penguin = penguin.drop(columns=['sex','species','island'])
    penguin,mean,std = get_norm(penguin)
    return penguin,mean,std

def normalize(dataset,mean,std):
    """
    Normalises the dataset by mean and standard deviation
    """
    mid = dataset - mean
    new_data = mid / std
    return new_data

def unnormalize(dataset,mean,std):
    """
    Reverts the normalised dataset to original format
    """
    df = pandas.DataFrame(dataset)
    mid = df*std
    original = mid + mean
    return original

def get_norm(data):
    """
    Provides the mean and standard deviation for the dataset so it can be normalised. 
    """
    mean = data.mean()
    std = data.std()
    data = normalize(data,mean,std)
    return data.to_numpy('float'),mean.to_numpy('float'),std.to_numpy('float')
