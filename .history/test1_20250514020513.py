import torch
print(torch.version.cuda)
print(torch.__version__)
torch.cuda.empty_cache()
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1e6:.2f} MB")
