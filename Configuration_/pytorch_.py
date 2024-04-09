import torch

print(torch.cuda.is_available())  # Check if CUDA is available
print(torch.backends.cudnn.enabled)  # Check if cuDNN is enabled
