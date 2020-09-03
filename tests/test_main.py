import subprocess
import os
from randomdatagen import generate_random_testing_data
import numpy
import ganrunner


def test_help():
    exp_output = b"""Usage: __main__.py [OPTIONS]\n
  This code creates and trains a GAN. Core elements of this code are sourced
  directly from the David Foster book 'Deep generative models' and have been
  modified to work with a numeric database such as the iris datasets taken
  from the library 'sklearn'. Soon this GAN should work on the DCWW dataset.

Options:
  --mode TEXT       mode?(s)pyder/(n)ormal/(m)arathon)
  --filepath TEXT   enter the file name and location of the database and model
  --epochs TEXT     choose how long that you want to train
  --dataset TEXT    choose the dataset/table that the GAN will train on - this
                    can't be a single letter - don't add .db

  --model TEXT      choose which model you what to use
  --opti TEXT       choose the optimiser you want to use
  --noise TEXT      choose the length of the noise vector
  --batch TEXT      choose how many fake data you want to make in one go
  --layers TEXT     choose the number of layers of each network
  --clip TEXT       if using WGAN choose the clipping threshold
  --core INTEGER    select the number of cores that you would like to run
  --sample INTEGER  choose the number of generated data you want:
                    (samples*batch)

  --help            Show this message and exit.
"""
    currunt_output = subprocess.run(
        ["python", "-m", "ganrunner", "--help"], capture_output=True
    )
    assert currunt_output.stdout == exp_output


def test_normal_run():
    generate_random_testing_data(50)
    file_size = os.stat("flight.db").st_size
    status = subprocess.run(
        [
            "python",
            "-m",
            "ganrunner",
            "--model=g",
            "--filepath=flight",
            "--opti=adam",
            "--noise=50",
            "--batch=50",
            "--layers=3",
            "--epochs=10",
            "--dataset=readings",
        ]
    )
    assert status.returncode == 0
    assert os.path.isfile("flight_model.h5")
    assert os.path.isfile("flight_generator.h5")
    assert os.path.isfile("flight_discriminator.h5")
    assert os.path.isfile("flight_parameters.npy")
    assert file_size < os.stat("flight.db").st_size and (file_size > 0)


def test_marathon():
    filepath = "marathon_test"
    layers = 2
    nodes = 7
    data = 3
    batch_size = 2
    noise_vector = 5
    epochs = 1020
    dataset = numpy.array([[1.0, 1.2, 1.3], [2.1, 2.2, 2.3]])
    testgan = ganrunner.dataGAN("adam", noise_vector, data, nodes, layers)
    untrained = testgan.discriminator.predict(dataset)
    ganrunner.marathon_mode(
        testgan, dataset, batch_size, noise_vector, filepath, epochs
    )
    trained = testgan.discriminator.predict(dataset)
    assert all(untrained < trained)
