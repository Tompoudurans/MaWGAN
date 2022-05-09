import math
import numpy as np
import ganrunner.tools as tools
import ganrunner.gans as gans
import logging
import os


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
    core,
    sample,
    rate,
    graph,
):
    """
    This code creates and trains a GAN.
    Core elements of this code are sourced directly from the David Foster book 'Deep generative models' and have been modified to work with a numeric database such as the iris datasets taken from the library 'sklearn'.
    Soon this GAN should work on the DCWW dataset.
    """
    click.echo("loading...")
    if graph == "True":
        graph = True
    else:
        graph = False
    if core != 0:
        tools.set_core(core)
    filename, extention = filepath.split(".")
    if extention == "csv":
        dataset = True
    tools.setup_log(filename + "_progress.log")
    parameters_list = [dataset, model, opti, noise, batch, layers, lambdas, rate]
    parameters, successfully_loaded = parameters_handeling(filename, parameters_list)
    epochs = int(epochs)
    try:
        database, details = load_data(filepath, extention)
    except tools.sqlman.sa.exc.OperationalError as oe:
        logging.exception(oe)
        try:
            table = tools.all_tables(filepath)
            print(dataset, "does not exists, try:")
            print(str(table))
        except Exception as e:
            print("file not found")
            logging.exception(e)
        os.remove(filename + "_parameters.npy")
        return
    except Exception as e:
        print("Data could not be loaded propely see logs for more info")
        logging.exception(e)
        return
    thegan, success = run(
        filename, epochs, parameters, successfully_loaded, database, batch
    )
    fake = None
    if success:
        fake = make_samples(
            thegan, database, sample, filename, details, extention, graph,
        )
    return thegan, fake


def unpack(p):
    """
    Unpacks the parameters
    """
    return p[1], p[2], int(p[3]), int(p[4]), int(p[5])


def setup(parameters_list):
    """
    Creates new parameters
    """
    parameters = []
    questions = [
        "table? ",
        "model? (linear/RNN/LSNM/GRU/GRUI)",
        "opti? ",
        "noise size? ",
        "batch size? ",
        "layers? ",
        "learning constiction? ",
        "rate? ",
    ]
    for q in range(len(questions)):
        if parameters_list[q] != None:
            param = parameters_list[q]
        else:
            if q < 3:
                param = input(questions[q])
            elif q < 6:
                param = input_int(questions[q])
            else:
                param = input_float(questions[q])
        parameters.append(param)
    return parameters


def input_int(question):
    """
    makes sure a number is inputed
    """
    while True:
        try:
            a = input(question)
            answer = int(a)
        except Exception:
            print("must be a number")
            if a == "" or None:
                raise RuntimeError
        else:
            return answer


def input_float(question):
    """
    makes sure a number is inputed
    """
    while True:
        try:
            a = input(question)
            answer = float(a)
        except Exception:
            print("must be a number")
            if a == "" or None:
                raise RuntimeError
        else:
            return answer


def load_data(filepath, extention):
    """
    Loads a dataset from an sql table
    """
    raw_data = tools.pd.read_csv(filepath)
    database, details = tools.procsses_sql(raw_data)
    return database, details


def load_gan_weight(filepath, mygan):
    """
    Loads weight from previous trained GAN
    """
    try:
        mygan.load_model(filepath)  # -----------------------------------
    except OSError:  # as 'Unable to open file':
        print(filepath)
    finally:
        return mygan


def create_model(parameters, no_field):
    """
    Builds the GAN using the parameters
    """
    lr = float(parameters[7])
    use_model, opti, noise_dim, batch, number_of_layers = unpack(parameters)
    try:
        base, network = use_model.split("-")
    except Exception:
        network = use_model
        base = "None"
    print("network:", network, "| base:", base)
    if base == "vgan":
        mygan = gans.vgan(
            optimiser="adam",
            input_dim=no_field,
            net_dim=noise_dim,
            number_of_layers=number_of_layers,
            lambdas=float(parameters[6]),
            learning_rate=lr,
            network=network,
        )
    else:
        mygan = gans.wGANgp(
            optimiser="adam",
            input_dim=no_field,
            net_dim=noise_dim,
            number_of_layers=number_of_layers,
            lambdas=float(parameters[6]),
            learning_rate=lr,
            network=network,
        )
    return mygan


def make_samples(mygan, database, batch, filepath, details, extention, show=True):
    """
    Creates a number of samples
    """
    if show:
        database = tools.pd.DataFrame(database)
        database = database.sample(batch)
    mean, std, info, col = details
    fullset = None
    generated_data = mygan.create_fake(batch)
    if show:
        tools.calculate_fid(generated_data, database.sample(batch))
    generated_data = tools.unnormalize(generated_data, mean, std)
    generated_data.columns = col
    if show:
        database = tools.unnormalize(database, mean, std)
        database.columns = col
        tools.dagplot(generated_data, database, filepath + "_" + str(s))
        print("plot")
    fullset = tools.decoding(generated_data, info)
    if extention == "db":
        tools.save_sql(fullset, filepath + ".db")
    elif extention == "csv":
        fullset.to_csv(filepath + "_synthetic.csv", index=False)
    else:
        print("I", end="")
    return fullset


def save_parameters(parameters, filepath):
    """
    Saves the parameters for the GAN
    """
    fname = filepath + "_parameters.npy"
    parameter_array = np.array(parameters)
    np.save(fname, parameter_array)


def load_parameters(filepath):
    """
    Loads the parameters for the GAN
    """
    try:
        parameter_array = np.load(filepath + "_parameters.npy", allow_pickle=True)
    except OSError:
        print("\n", filepath, "does not exists")
        successfully_loaded = False
        parameter_array = None
    else:
        successfully_loaded = True
    return parameter_array, successfully_loaded


def parameters_handeling(filepath, parameters_list):
    """
    Load parameters if they exist, otherwise saves new ones
    """
    parameters, successfully_loaded = load_parameters(filepath)
    if not successfully_loaded:
        parameters = setup(parameters_list)
        print(parameters)
        save_parameters(parameters, filepath)
    return parameters, successfully_loaded


def run(
    filepath, epochs, parameters, successfully_loaded, database, batch, usegpu=False
):
    """
    Creates and trains a GAN from the parameters provided.
    It will load the weights of the GAN if they exist.
    An option will be given to create samples.
    """
    # select dataset
    logging.info(filepath)
    no_field = len(database[1])
    try:
        mygan = create_model(parameters, no_field)
    except Exception as e:
        print("building failed, check you parameters")
        logging.error("building failed due to " + str(e))
        return None, False
    if successfully_loaded:
        mygan = load_gan_weight(filepath, mygan)
    if epochs > 0:
        step = int(math.ceil(epochs * 0.1))
        checkI = tools.pd.DataFrame(database)
        checkII = checkI.isnull().sum().sum() > 0
        try:
            mygan.train(database, batch, epochs, checkII, step, 15, usegpu)
        except Exception as e:
            logging.exception(e)
            print("training failed check you parameters")
            return None, False
        else:
            mygan.save_model(filepath)
    return mygan, True
