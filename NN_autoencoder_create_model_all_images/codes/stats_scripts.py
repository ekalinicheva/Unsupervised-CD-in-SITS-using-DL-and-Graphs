import sys
import torch
from subprocess import check_output
import matplotlib.pyplot as plt


# Check if it GPU with memory > 4GB is available
def on_gpu():
    gpu = False
    if torch.cuda.is_available():
        try:
            gpu_index, gpu_name, memory_total, memory_free = check_output(
                ["C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe", "--query-gpu=index,name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"]).decode(sys.stdout.encoding).strip().split(",")
            memory_free = int(memory_free)
            if memory_free >= 4000:
                gpu = True
        except:
            gpu = True
    return gpu


# Plot loss for each epoch
def plotting(epochs, loss_array, path):
    fig, ax = plt.subplots()
    ax.plot(range(1, epochs+1), loss_array)
    # ax.set_xticks(range(0, epochs+1, 50))
    ax.set(xlabel='epochs', ylabel='loss',
           title=('Loss epochs ='+ str(epochs)))
    ax.grid()
    fig.savefig(path+'Loss_epochs_'+str(epochs)+'.png')
    plt.close(fig)


# Print stats about current calculations to console and to file
def print_stats(stats_file, text, print_to_console=True):
    with open(stats_file, 'a') as f:
        if isinstance(text, list):
            for t in text:
                f.write(t + "\n")
                if print_to_console:
                    print(t)
        else:
            f.write(text + "\n")
            if print_to_console:
                print(text)
    f.close()