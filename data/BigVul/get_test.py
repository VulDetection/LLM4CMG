import pandas as pd

# 读取原始Excel文件
df = pd.read_excel('BigVul_all.xlsx')

# 随机抽取200条数据（不重复）
sample_df = df.sample(n=200, random_state=42)  # random_state 保证可重复性

# 保存为新的Excel文件
sample_df.to_excel('BigVul_test.xlsx', index=False)

print("✅ 随机抽取完成，已保存为 BigVul_test.xlsx")