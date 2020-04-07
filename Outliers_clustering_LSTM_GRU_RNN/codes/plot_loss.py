import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plotting(epochs, loss_array, path):

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(loss_array)+1), loss_array)
    # ax.set_xticks(range(0, epochs+1, 50))
    ax.set(xlabel='epochs', ylabel='loss',
           title=('Loss epochs ='+ str(epochs+1)))
    ax.grid()
    fig.savefig(path+'Loss_epochs_'+str(epochs+1)+'.png')
    plt.close(fig)