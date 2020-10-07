import main
import numpy


def test_load_data():
    loaded_data = load_data(sets, "test.db")
    dataset = numpy.array([[1.0, 1.2, 1.3], [2.1, 2.2, 2.3]])
    assert loaded_data == dataset


def test_show_samples():
    pass


def test_parameters_handeling():
    # saves
    saved_parameters, successfully_loaded = parameters_handeling(
        "tests/testing", parameters_list
    )
    # loads
    loaded_parameters, successfully_loaded = parameters_handeling("tests/testing", None)
    assert loaded_parameters == saved_parameters
