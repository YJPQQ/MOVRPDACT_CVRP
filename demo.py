import torch

depot = torch.randint(low = 0, high=10, size = (2,))
print(depot)

repeat_depot = depot.view(-1, 2).repeat(10,1)
print(repeat_depot)