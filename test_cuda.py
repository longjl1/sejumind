import torch

print("PyTorch版本：", torch.__version__)
print("CUDA是否可用：", torch.cuda.is_available())
print("CUDA设备数量：", torch.cuda.device_count())

