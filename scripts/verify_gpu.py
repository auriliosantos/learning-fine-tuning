import torch

print("=== PyTorch GPU Verification ===")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    total_memory = torch.cuda.get_device_properties(current_device).total_memory
    memory_gb = total_memory / (1024 ** 3)
    
    print(f"Device Count: {device_count}")
    print(f"Current Device ID: {current_device}")
    print(f"Device Name: {device_name}")
    print(f"Total VRAM: {memory_gb:.2f} GB")
    
    # Run a quick tensor computation on the GPU
    torch.manual_seed(42)
    x = torch.rand(3, 3).cuda()
    y = torch.rand(3, 3).cuda()
    z = x @ y
    print("Test Matrix Multiplication: Successful!")
else:
    print("❌ GPU not detected. PyTorch is running in CPU mode.")
