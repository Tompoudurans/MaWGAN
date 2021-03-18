# -*- coding: utf-8 -*-

import ganrunner


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
    return thegan, fake, details

def fid_run(batch,per):
    epochs = 30000
    a, b, c = main(
        None,
        "Kag.csv",
        epochs,
        "wgangp",
        "adam",
        batch,
        batch,
        5,
        10,
        4,
        0.00001
    )
    full = ganrunner.tools.pd.read_csv("kag2.csv")
    x = ganrunner.tools.get_norm(full)
    original = ganrunner.tools.pd.DataFrame(x[0])
    for i in range(20):
        generated_data = ganrunner.tools.pd.DataFrame(a.create_fake(batch))
        gendata = ganrunner.decoding(generated_data,c[2])
        ganrunner.tools.calculate_fid(gendata, original.sample(batch))
    return original, generated_data, c
i=5
gan,data,de = fid_run(250, str(i))
