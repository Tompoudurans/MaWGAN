import pandas
import numpy.nan

def normalize(dataset, mean, std):
    """
    Normalises the dataset by mean and standard deviation
    """
    mid = dataset - mean
    new_data = mid / std
    return new_data


def unnormalize(dataset, mean, std):
    """
    Reverts the normalised dataset to original format
    """
    df = pandas.DataFrame(dataset)
    mid = df * std
    original = mid + mean
    return original


def get_norm(data):
    """
    Provides the mean and standard deviation for the dataset so it can be normalised.
    """
    mean = data.mean()
    std = data.std()
    data = normalize(data, mean, std)
    return data.to_numpy("float"), mean.to_numpy("float"), std.to_numpy("float")

def missing_numb(data,col,method):
    if method == "drop":
        return data.dropna()
    elif method == "zero":
        return data.replace(numpy.nan,0)
    elif method == "mean":
        for i in col:
            data[i]=data[i].replace(numpy.nan,data[i].mean())
        return data
    else:
        return data