import categorical
import pandas

data = pandas.read_csv('penguins_size.csv')
print(data)
new,bit = categorical.encoding(data)

stuff = categorical.decoding(new,bit)
print(stuff)
