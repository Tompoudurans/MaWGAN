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
        for value in range(set_of_cat.shape[0]):
            restore.append(set_of_cat.iloc[value].idxmax())
        data[details[current][0]] = restore
        current = current + 1
        position = end
    data = data.drop(columns=data.columns[range(start, col_len)])
    return data

def binary_in(data):
    tag = []
    for i in range(data.shape[1]):
        column = data.iloc[:,i]
        if column.max() == 1 and column.min() == 0:
            data.iloc[:,i] = column.fillna(0.5)
            tag.append(True)
        else:
            tag.append(False)
    return data,tag

def binary_out(data,tag):
    for i in range(data.shape[1]):
        if tag[i]:
            data.iloc[:,i] = data.iloc[:,i].round()
    return data
