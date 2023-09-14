# import numpy as np
# import torch as tor
# import matplotlib.pyplot as plt
# import torchsummary as ts
# from torchvision import transforms
# from torchvision.datasets import MNIST


# idx = int(input("Please enter a number from 0 to 59999: "))

# train_transform = transforms.Compose([transforms.ToTensor()])
# train_set = MNIST('./data/mnist', train = True, download = True, transform = train_transform)


# plt.imshow(train_set.data[idx], cmap='gray')
# print(train_set.targets[idx])
# plt.show()


import torch

# Check if GPU (CUDA) is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    print("GPU(s) available:")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
else:
    print("No GPU available. PyTorch will use CPU.")
