
# Pyhton code of MaWGAN

## About

This is the code for "MaWGAN: a Generative Adversarial Network to create synthetic
data from datasets with missing data"
https://orca.cardiff.ac.uk/id/eprint/148018/1/electronics-11-00837.pdf
If you use this code, please cite:
```
@article{poudevigne2022mawgan,
  title={MaWGAN: a generative adversarial network to create synthetic data from datasets with missing data},
  author={Poudevigne-Durance, Thomas and Jones, Owen Dafydd and Qin, Yipeng},
  journal={Electronics},
  volume={11},
  number={6},
  pages={837},
  year={2022},
  publisher={MDPI}
}
```
Acknowledgements: We would like to thank Green9 for open sourcing his excellent work on WGAN-GP: https://github.com/Zeleni9/pytorch-wgan, which inspired this project."

## Installation

### Basic software requirements

To install, the following is assumed to be available:

- Python 3.8
- A recent version of git.

### To install

The following commands will download and install the software

    $ git clone https://github.com/Tompoudurans/MaWGAN
    $ cd MaWGAN
    $ python setup.py develop


To confirm installation was successful run the following:

    $ python -m Mawgan --help

### Test

The tool is tested using the Python testing framework pytest. To install pytest:

    $ python -m pip install pytest

To run the tests:    

    $  python -m pytest


## To run the code

$ python -m Mawgan <Options>

you only need to put the filename and the epochs to read a pre-trained model. if put the epochs to 0 it will skip training phase of the code.


### Options
    "--test"          none     "test the instalment"
    "--filepath"      text     "enter the file name and location of the database and model"
    "--epochs"        int      "choose how long that you want to train"
    "--dataset"       text     "choose the dataset/table that the GAN will train on"
    "--opti"          text     "choose the optimiser you want to use"
    "--nodes"         int      "choose the number nodes per layer"
    "--batch"         int      "choose how many datapoints is process when traing in one go"
    "--layers"        int      "choose the number of layers of each network"
    "--lambdas",      int      "learning penalty"
    "--sample"        int      "choose the number of generated data"
    "--rate"          float    "choose the learning rate of the model"
    "--usegpu"        1/0      "set to 1 to use gpu"

### data format

this code accept both .db files and .csv files, missing values should be empty
V [5,,4,2]
X [5,NA,4,2]
X [5,-,4,2]
the data must have columns names but no index column, the data can be numerical or/and categorical. 

examples:
.db file without data missing, and displaying a compare graph

    $ python -m Mawgan --filepath=iris.db --opti=adam --nodes=100 --batch=100 --layers=5 --epochs=1000 --dataset=all --rate=0.0001 --lambdas=10 --sample=300

.csv file with data missing

    $ python -m Mawgan --filepath=10_Deprivation_percent.csv --opti=adam --nodes=200 --batch=300 --layers=5 --epochs=1000 --rate=0.0001 --lambdas=10 --sample=200

##file discription

src/.../gans/Main_model.py contain the main model
src/.../gans/masker.py contain the masking algorithm
src/.../tools/fid.py functions to calculate FID
src/.../tools/sqlman.py read and write .db files
src/.../tools/preprocess.py data ready for training
test/ code to tests files in src with pytest
env/ is the mass testing for the paper, done by running master.py
