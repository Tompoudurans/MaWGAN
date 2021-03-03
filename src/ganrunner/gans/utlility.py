from torch import tensor

def make_mask(data):
    """
    make mask hidding missing data
    """
    binary_mask = data.isnan()
    inverse_mask = tensor(binary_mask,dtype=int)
    mask = 1-inverse_mask
    return mask,binary_mask

def copy_format(template,data):
    """
    create a mask from the template and apply it to the data
    """
    mask,binary_mask = make_mask(template)
    masked_data = data*mask
    template[binary_mask] = 0
    return template, masked_data
