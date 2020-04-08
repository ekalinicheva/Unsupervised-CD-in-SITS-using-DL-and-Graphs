import sys
import torch
from subprocess import check_output

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