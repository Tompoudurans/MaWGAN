import math
import numpy as np
import ganrunner.tools as tools
import ganrunner.gans as gans
import click
import logging
import os


@click.command()
@click.option(
    "--filepath",
    prompt="filepath? ",
    help=" enter the file name and location of the database and model",
)
@click.option(
    "--epochs", prompt="epochs? ", help="choose how long that you want to train"
)
@click.option(
    "--dataset",
    default=None,
    help="choose the dataset/table that the GAN will train on",
)
@click.option("--model", default=None, help="choose which model you what to use")
@click.option("--opti", default=None, help="choose the optimiser you want to use")
@click.option("--noise", default=None, help="choose the length of the noise vector")
@click.option(
    "--batch", default=None, help="choose how many fake data you want to make in one go"
)
@click.option(
    "--layers", default=None, help="choose the number of layers of each network"
)
@click.option("--lambdas", type=float, default=None, help="learning penalty")
@click.option(
    "--core",
    default=0,
    type=int,
    help="select the number of cores that you would like to run",
)
@click.option(
    "--sample",
    default=1,
    type=int,
    help="choose the number of generated data you want: (samples*batch)",
)
@click.option(
    "--rate",
    default=None,
    type=float,
    help="choose the learing rate of the model",
)
@click.option(
    "--graph",
    default=True,
    type=bin,
    help="compare sythetic and on a grath, not sutable for large dataset",
)
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
        database, details = load_data(parameters[0], filepath, extention)
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
    thegan, success = run(filename, epochs, parameters, successfully_loaded, database)
    fake = None
    if success:
        fake = make_samples(
            thegan,
            database,
            int(parameters[4]),
            sample,
            filename,
            details,
            extention,
            graph,
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


def load_data(sets, filepath, extention):
    """
    Loads a dataset from an sql table
    """
    if extention == "db":
        raw_data = tools.load_sql(filepath, sets)
    else:
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
        print("file not found, starting from scratch")
    finally:
        return mygan


def create_model(parameters, no_field):
    """
    Builds the GAN using the parameters
    """
    lr = float(parameters[7])
    use_model, opti, noise_dim, batch, number_of_layers = unpack(parameters)
    mygan = gans.wGANgp(
        optimiser="adam",
        input_dim=no_field,
        noise_size=noise_dim,
        batch_size=batch,
        number_of_layers=number_of_layers,
        lambdas=float(parameters[6]),
        learning_rate=lr,
        network=use_model
    )
    mygan.summary()
    return mygan, batch, noise_dim


def make_samples(
    mygan, database, batch, samples, filepath, details, extention, show=True
):
    """
    Creates a number of samples
    """
    database = tools.pd.DataFrame(database)
    database = database.sample(batch)
    mean, std, info, col = details
    fullset = None
    for s in range(int(samples)):
        generated_data = mygan.create_fake(batch)
        if s == 0 and show:
            tools.calculate_fid(generated_data, database.sample(batch))
        generated_data = tools.unnormalize(generated_data, mean, std)
        generated_data.columns = col
        print("unnorm complete gen")
        if s == 0 and show:
            database = tools.unnormalize(database, mean, std)
            database.columns = col
            print("unnorm complete org")
        if show:
            tools.dagplot(generated_data, database, filepath + "_" + str(s))
            print("plot")
        values = tools.decoding(generated_data, info)
        print("sample", s)
        if s > 0:
            fullset = tools.pd.merge(fullset, values, "outer")
        else:
            fullset = values
    if extention == "db":
        tools.save_sql(fullset, filepath + ".db")
    else:
        fullset.to_csv(filepath + "_synthetic.csv", index=False)
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
        print("file not found, starting from scratch")
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


def run(filepath, epochs, parameters, successfully_loaded, database):
    """
    Creates and trains a GAN from the parameters provided.
    It will load the weights of the GAN if they exist.
    An option will be given to create samples.
    """
    # select dataset
    no_field = len(database[1])
    try:
        mygan, batch, noise_dim = create_model(parameters, no_field)
    except Exception as e:
        print("building failed, check you parameters")
        logging.error("building failed due to" + str(e))
        return None, False
    if successfully_loaded:
        mygan = load_gan_weight(filepath, mygan)
    if epochs > 0:
        step = int(math.ceil(epochs * 0.1))
        checkI = tools.pd.DataFrame(database)
        checkII = checkI.isnull().sum().sum() > 0
        try:
            mygan.train(database, batch, epochs, checkII, step)
        except Exception as e:
            logging.exception(e)
            print("training failed check you parameters")
            return None, False
        else:
            mygan.save_model(filepath)
            # tools.show_loss_progress(mygan.d_losses, mygan.g_losses, filepath)
    return mygan, True


if __name__ == "__main__":
    main()
