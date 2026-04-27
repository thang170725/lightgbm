import pandas as pd

df = pd.DataFrame({
    "name": ['thang', 'minh', 'hieu', 'long', 'linh', 'trang'],
    "age": [12,7,8,14,5,8]
})

ratio = (df == 0).mean()
print(ratio)