import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("✅ CUDA is available!")
    print("GPU Device Name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
else:
    print("❌ CUDA not available, running on CPU.")
