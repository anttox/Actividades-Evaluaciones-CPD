import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim

# Funcion para configurar el proceso de comunicacion
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# Funcion para limpiar el proceso de comunicacion
def cleanup():
    dist.destroy_process_group()

# Definicion del modelo
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

# Funcion de demostracion basica de DDP
def demo_basic(rank, world_size):
    setup(rank, world_size)
    
    # Comprobar si el numero de GPUs es suficiente
    available_gpus = torch.cuda.device_count()
    if available_gpus < world_size:
        print(f"No hay suficientes GPUs. Se requieren {world_size}, pero solo hay {available_gpus} disponibles.")
        return
    
    # Asignar dispositivo CUDA basado en el rank
    device = torch.device(f'cuda:{rank}')
    model = ToyModel().to(device)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    dataset = torch.randn(20, 10)
    targets = torch.randn(20, 5)

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = ddp_model(dataset.to(device))
        loss = loss_fn(outputs, targets.to(device))
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"Epoca {epoch+1}, Perdida: {loss.item()}")

    cleanup()

# Funcion para ejecutar la demostracion con multiples procesos
def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    # Comprobar cuantas GPUs estan disponibles
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("No hay GPUs disponibles.")
    elif available_gpus < 2:
        print(f"Solo hay {available_gpus} GPU(s) disponible(s). Ejecutando en una sola GPU.")
        world_size = 1
        run_demo(demo_basic, world_size)
    else:
        print(f"Se encontraron {available_gpus} GPUs. Ejecutando en {available_gpus} GPUs.")
        world_size = 2
        run_demo(demo_basic, world_size)
