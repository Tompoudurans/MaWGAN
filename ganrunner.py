from sklearn import datasets
from tengan import dataGAN

iris = datasets.load_wine()
mygan = dataGAN('adam',13,13)
mygan.train(iris.data,13,1000)
#noise = np.random.normal(0, 1, (100, 100))
#print(mygan.generator.predict(noise))
