import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('BigVul_new_recom.csv', encoding='utf-8')

# 将 DataFrame 保存为 Excel 文件
df.to_excel('BigVul_new_recom.xlsx', index=False)