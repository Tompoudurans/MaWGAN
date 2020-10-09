def factorizing(data):
    """
    Transforms categorical data into numerical, saves maping on a list.
    """
    details
    for name in data.columns:
        if "O" == data[name].dtype:
            new = pandas.get_dummies(data[name])
            data[new.columns] = new
            data.drop(columns = name)
            details.append([name,len(new.columns)])
    return data
