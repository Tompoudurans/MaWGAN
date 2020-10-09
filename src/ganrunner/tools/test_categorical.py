import categorical
import pandas

data = pandas.read_csv('penguins_size.csv')
new,bit = categorical.encoding(data)
print('new\n',new,'\nlist ',bit)
