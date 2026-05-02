"""Small CUDA smoke test for SoftComp development environments."""

from __future__ import annotations

import torch


def main() -> None:
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.randn(256, 256, device=device)
        y = x @ x.T
        torch.cuda.synchronize()
        print(f"device={torch.cuda.get_device_name(device)} norm={float(y.norm().cpu()):.6f}")
    else:
        x = torch.randn(128, 128)
        y = x @ x.T
        print(f"cpu_norm={float(y.norm()):.6f}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
