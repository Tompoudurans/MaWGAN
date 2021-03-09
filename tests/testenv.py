# -*- coding: utf-8 -*-

import ganrunner

epochs = 12000


def main(
    dataset,
    filepath,
    epochs,
    model,
    opti,
    noise,
    batch,
    layers,
    lambdas,
    sample,
    rate,
):
    """
    This code creates and trains a GAN.
    Core elements of this code are sourced directly from the David Foster book 'Deep generative models' and have been modified to work with a numeric database such as the iris datasets taken from the library 'sklearn'.
    Soon this GAN should work on the DCWW dataset.
    """
    filename, extention = filepath.split(".")
    if extention == "csv":
        dataset = True
    ganrunner.tools.setup_log(filename + "_progress.log")
    parameters_list = [dataset, model, opti, noise, batch, layers, lambdas, rate]
    parameters, successfully_loaded = ganrunner.parameters_handeling(
        filename, parameters_list
    )
    epochs = int(epochs)
    database, details = ganrunner.load_data(parameters[0], filepath, extention)
    thegan, success = ganrunner.run(
        filename, epochs, parameters, successfully_loaded, database
    )
    fake = None
    if success:
        fake = ganrunner.make_samples(
            thegan,
            database,
            int(parameters[4]),
            sample,
            filename,
            details,
            extention,
            False,
        )
    return thegan, fake


a, b = main(
    None,
    "missiris.csv",
    epochs,
    "wgangp",
    "adam",
    150,
    150,
    5,
    10,
    3,
    0.0004,
)
full = ganrunner.tools.pd.read_csv("fulliris.csv")
x = ganrunner.tools.get_norm(full)
y = ganrunner.tools.get_norm(b)
ganrunner.calculate_fid(x[0], y[0])
