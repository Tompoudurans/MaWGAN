import subprocess
import os
from randomdatagen import generate_random_testing_data
import numpy
import Mawgan


def test_gen_rand():
    if not os.path.isfile("flight.db"):
        generate_random_testing_data(20)


def test_break_set():
    status = subprocess.run(
        [
            "python",
            "-m",
            "Mawgan",
            "--filepath=flight.db",
            "--opti=adam",
            "--nodes=60",
            "--batch=60",
            "--layers=2",
            "--epochs=10",
            "--dataset=rando",
            "--rate=0.1",
            "--lambdas=10",
        ]
    )
    assert status.returncode == 0
    os.remove("flight_parameters.npy")


def test_break_bulid():
    status = subprocess.run(
        [
            "python",
            "-m",
            "Mawgan",
            "--filepath=flight.db",
            "--opti=adam",
            "--nodes=60",
            "--batch=sixty",
            "--layers=3",
            "--epochs=10",
            "--dataset=readings",
            "--rate=0.1",
            "--lambdas=10",
        ]
    )
    assert status.returncode == 0
    os.remove("flight_parameters.npy")


def test_normal_run():
    file_size = os.stat("flight.db").st_size
    status = subprocess.run(
        [
            "python",
            "-m",
            "Mawgan",
            "--filepath=flight.db",
            "--opti=adam",
            "--nodes=30",
            "--batch=20",
            "--layers=3",
            "--epochs=10",
            "--dataset=readings",
            "--rate=0.001",
            "--lambdas=10",
            "--sample=20",
        ]
    )
    assert status.returncode == 0
    assert os.path.isfile("flight_generator.pkl")
    assert os.path.isfile("flight_critic.pkl")
    assert os.path.isfile("flight_parameters.npy")


def test_reload():
    status = subprocess.run(
        [
            "python",
            "-m",
            "Mawgan",
            "--epochs=10",
            "--filepath=flight.db",
        ]
    )
    assert status.returncode == 0
    # assert len(status.stdout) > 12
