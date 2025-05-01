import torch, platform
print("torch", torch.__version__, "python", platform.python_version())
print("mps available:", torch.backends.mps.is_available())