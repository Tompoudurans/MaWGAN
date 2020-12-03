from sklearn import datasets
from wgan_gradient_penalty import WGAN_GP
from dataman import dagplot
import torch
gan = WGAN_GP("optimiser",4,150,150,5,10,0.0001)
            #input_dim,
            #noise_size,
            #batch_size,
            #number_of_layers,
            #lambdas,
            #learning_rate)
iris = datasets.load_iris()
data = iris.data

gan.train(data,150,1000)

z = gan.get_torch_variable(torch.randn(gan.batch_size, 150))
x = gan.G(z)
print(x)
dagplot(x.detach().numpy(), data, "thing")
