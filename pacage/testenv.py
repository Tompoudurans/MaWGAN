# -*- coding: utf-8 -*-

import ganrunner
import multiprocessing as mpg

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

def fid_run(block):
    per,dataname,batch = block
    epochs = 20000
    a, b, c= main(
        None,
        str(per) + "0" + dataname,
        epochs,
        "wgangp",
        "adam",
        batch,
        batch,
        5,
        10,
        1,
        0.0004,
    )
    full = ganrunner.tools.pd.read_csv("00" + dataname)
    x = ganrunner.tools.get_norm(full)
    original = ganrunner.tools.pd.DataFrame(x[0])
    fids = []
    for i in range(20):
        generated_data = ganrunner.tools.pd.DataFrame(a.create_fake(batch))
        gendata = ganrunner.decoding(generated_data,c[2])
        fids.append(ganrunner.tools.calculate_fid(gendata, original.sample(batch)))
    return fids

def poolrun(dataname,batch):
    block = []
    for i in range(6):
        block.append([i,dataname,batch])
    p = mpg.Pool(6)
    result = p.map(fid_run,block)
    p.close()
    p.join()
    return result

def singal(dataname,batch):
    set = []
    for i in range(6):
        set.append(fid_run([i,dataname,batch]))
    return set

def one_dataset(dataname,muti,batch):
    if muti:
        fidata = poolrun(dataname,batch)
    else:
        fidata = singal(dataname,batch)
    frame = ganrunner.tools.pd.DataFrame(fidata)
    frame.to_csv("fids" + dataname)

if __name__ == '__main__':
    muti = True
        one_dataset(datanames[1],muti,200)
