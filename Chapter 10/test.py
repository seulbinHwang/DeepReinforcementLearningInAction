import einops # 내장 einsum 함수를 확장하는 Einops라는 패키지를 활용
import torch

x = torch.randn(5, 49 ,3)
answer = einops.rearrange(x, "batch (h w) c -> batch h w c", h=7).shape
print(answer)