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
    #generate_random_testing_data(50)
    file_size = os.stat("flight.db").st_size
    status = subprocess.run(
        [
            "python",
            "-m",
            "ganrunner",
            "--model=gan",
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
