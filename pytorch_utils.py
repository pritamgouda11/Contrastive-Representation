import torch
import numpy as np


# 'cuda' device for supported nvidia GPU, 'mps' for Apple silicon (M1-M3)
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'mps'\
        if torch.backends.mps.is_available() else 'cpu')
def from_numpy(
        x: np.ndarray,
        dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    # raise NotImplementedError('Convert numpy array to torch tensor here and send to device')
    return torch.from_numpy(x).to(device, dtype=dtype)
    # print(f"Tensor: {tensor}, Device: {tensor.device}")

def to_numpy(x: torch.Tensor) -> np.ndarray:
    # raise NotImplementedError('Convert torch tensor to numpy array here')
    return x.detach().cpu().numpy()
# converted_np_array = to_numpy(tensor)
# print(f"Numpy Array: {converted_np_array}")
# np_array = np.array([1, 2, 3], dtype=np.float32)
