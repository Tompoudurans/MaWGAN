import pandas
import numpy

def encoding(data):
    """
    Transforms categorical data into numerical, saves maping on a list.
    """
    details = [len(data.columns)]
    for name in data.columns:
        if "O" == data[name].dtype:
            new = pandas.get_dummies(data[name])
            data[new.columns] = new
            data = data.drop(columns = name)
            details.append([name,len(new.columns)])
    return data, details

def decoding(data,details):
    print('point')
    col_len = len(data.columns)
    position = details[0] - len(details) + 1
    start = position
    current = 1
    while position < col_len:
        end = position + details[current][1]
        set_of_cat = data.iloc[:,position:end]
        restore = []
        for value in range(set_of_cat.shape[0]):
            try:
                restore.append(set_of_cat.ix[value].idxmax())
            except TypeError:
                restore.append(None)
        data[details[current][0]] = restore
        current = current + 1
        position = end
    data = data.drop(:,start:end)
    return data
