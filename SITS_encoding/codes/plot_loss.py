import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plotting(epochs, loss_array, path):

    fig, ax = plt.subplots()
    ax.plot(range(1, epochs+1), loss_array)
    # ax.set_xticks(range(0, epochs+1, 50))
    ax.set(xlabel='epochs', ylabel='loss',
           title=('Loss epochs ='+ str(epochs)))
    ax.grid()
    fig.savefig(path+'Loss_epochs_'+str(epochs)+'.png')
    plt.close(fig)