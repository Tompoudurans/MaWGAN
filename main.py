from ganrunner import run
from ganrunner import parameters_handeling
import click
@click.command()
@click.option('--mode', default='n', help='mode?(s)pyder/(n)ormal/(m)arathon)')
@click.option('--filepath', prompt='filepath? ')
@click.option('--epochs', prompt='epochs? ')
@click.option('--dataset', default=None)
@click.option('--model', default=None)
@click.option('--opti', default=None)
@click.option('--noise', default=None)
@click.option('--batch', default=None)
@click.option('--layers', default=None)
@click.option('--clip', default=None)

def main(dataset,mode,filepath,epochs,model,opti,noise,batch,layers,clip):
    click.echo('loading...')
    parameters_list = [dataset,model,opti,noise,batch,layers,clip]
    parameters,successfully_loaded = parameters_handeling(filepath,parameters_list)
    epochs = int(epochs)
    run(mode,filepath,epochs,parameters,successfully_loaded)

if __name__ == '__main__':
    main()
