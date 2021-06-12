import torch

output = torch.rand([10, 2])
print(output)
_, predict = torch.max(output, 1)
print(predict)
