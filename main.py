from ganrunner import run
import click
@click.command()
@click.option('--mode', default='n', help='mode?(s)pyder/(n)ormal/(m)arathon)')
@click.option('--filepath', default='#')
@click.option('--epochs', prompt='epochs? ')

def maind(mode,filepath,epochs):
    epochs = int(epochs)
    click.echo('loading...')
    run(mode,filepath,epochs)

if __name__ == '__main__':
    maind()
