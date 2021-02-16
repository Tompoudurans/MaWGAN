import pandas

def encoding(data):
    """
    Transforms categorical data into numerical, saves maping on a list.
    """
    details = [len(data.columns)]
    for name in data.columns:
        if "O" == data[name].dtype:
            new = pandas.get_dummies(data[name])
            data[new.columns] = new
            data = data.drop(columns=name)
            details.append([name, len(new.columns)])
        if data[name].max() == 1 and data[name].min() == 0:
            new = data[name]
            data = data.drop(columns=name)
            data[name] = new
            details.append([name, 1])
    return data, details


def decoding(data, details):
    """
    Transforms numerical data into categorical data using the saved mapping (details)
    """
    col_len = len(data.columns)
    position = details[0] - len(details) + 1
    start = position
    current = 1
    while position < col_len:
        try:
            end = position + details[current][1]
        except IndexError:
            break
        set_of_cat = data.iloc[:, position:end]
        restore = []
        if set_of_cat.shape[1] == 1:
            data[details[current][0]] = set_of_cat.round()
        else:
            for value in range(set_of_cat.shape[0]):
                restore.append(set_of_cat.iloc[value].idxmax())
            data[details[current][0]] = restore
        current = current + 1
        position = end
    #data = data.drop(columns=data.columns[range(start, col_len)])
    return data
