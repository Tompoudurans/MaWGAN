import subprocess
import os
from randomdatagen import generate_random_testing_data
import numpy
import ganrunner


def test_break_set():
    status = subprocess.run(
        [
            "python",
            "-m",
            "ganrunner",
            "--model=gan",
            "--filepath=flight.db",
            "--opti=adam",
            "--noise=50",
            "--batch=50",
            "--layers=2",
            "--epochs=10",
            "--dataset=rando",
        ]
    )
    assert status.returncode == 0
    assert not os.path.isfile("flight_parameters.npy")


def test_break_bulid():
    generate_random_testing_data(50)
    status = subprocess.run(
        [
            "python",
            "-m",
            "ganrunner",
            "--model=gan",
            "--filepath=flight.db",
            "--opti=adam",
            "--noise=50",
            "--batch=fithy",
            "--layers=3",
            "--epochs=10",
            "--dataset=readings",
        ]
    )
    assert status.returncode == 0
    assert not os.path.isfile("flight_parameters.npy")


def test_normal_run():
    file_size = os.stat("flight.db").st_size
    status = subprocess.run(
        [
            "python",
            "-m",
            "ganrunner",
            "--model=gan",
            "--filepath=flight.db",
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
