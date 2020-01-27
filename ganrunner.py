from sklearn import datasets
from tengan import dataGAN

iris = datasets.load_iris()
mygan = dataGAN('adam',100,4)
mygan.train(iris.data,100,1000)
noise = np.random.normal(0, 1, (100, 100))
print(mygan.generator.predict(noise))
