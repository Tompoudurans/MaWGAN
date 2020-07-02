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
    col = penguin.columns    
    penguin,mean,std = get_norm(penguin)
    return penguin,mean,std,col

def normalize(dataset,mean,std):
    mid = dataset - mean
    new_data = mid / std
    return new_data

def unnormalize(dataset,mean,std):
    df = pandas.DataFrame(dataset)
    mid = df*std
    original = mid + mean
    return original

def get_norm(data):
    mean = data.mean()
    std = data.std()
    data = normalize(data,mean,std)
    return data.to_numpy('float'),mean.to_numpy('float'),std.to_numpy('float')
