import torch
import psutil

free_mem, total_mem = torch.cuda.mem_get_info(device=0)
print(f"Free GPU memory: {free_mem / 1024**3:.2f} GB")
print(f"Total GPU memory: {total_mem / 1024**3:.2f} GB")

print(f"Memory allocated by tensors: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"Memory reserved by caching allocator: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_devices}")
    for i in range(num_devices):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA devices available. Using CPU.")

mem = psutil.virtual_memory()
print(f"Total RAM: {mem.total / 1024**3:.2f} GB")
print(f"Available RAM: {mem.available / 1024**3:.2f} GB")