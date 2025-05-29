import numpy as np

one_d_array = np.array([1, 2, 3])
print(one_d_array)
# 方法1：reshape为1行3列
two_d_array = one_d_array.reshape(1, -1)
print(two_d_array.tolist())  # 输出：[[1, 2, 3]]
print(len(two_d_array[0])) 
print(two_d_array.size(0))

# 方法2：增加新维度（保持原始数据顺序）
two_d_array = one_d_array[np.newaxis, :]
print(two_d_array.tolist())  # 输出：[[1, 2, 3]]