import torch

x = torch.tensor([1,2,3,4,5])
y = torch.tensor([10,20,30,40,50])

# torch.where(condition, input, other, *, out=None) → Tensor
# out = input if condition is True, else out = other
condition = x > 3
result = torch.where(condition, x, y) # tensor([10, 20, 30,  4,  5])
print(result)    

t = torch.arange(0, 10, 2)
print(t)

# outer
v1 = torch.tensor([1, 2, 3])
v2 = torch.tensor([4, 5, 6])
res = torch.outer(v1, v2)
print(res)

# 3个维度 两个batch 每个batch 2行3列
t1 = torch.tensor([[[1,2,3],[4,5,6]],[[13,14,15],[16,17,18]]])
t2 = torch.tensor([[[7,8,9],[10,11,12]],[[19,20,21],[22,23,24]]])
res = torch.cat((t1, t2), dim=-1) # 在最后一维拼接
print(res)

t1 = torch.Tensor([1, 2, 3])
print(t1.size()) # torch.Size([3])
t2 = t1.unsqueeze(0) # 在第0维增加一个维度
print(t2.size()) # torch.Size([1, 3])