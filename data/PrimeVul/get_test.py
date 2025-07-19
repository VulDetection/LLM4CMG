import pandas as pd

# 读取原始Excel文件
df = pd.read_excel('PrimeVul_all.xlsx')

# 随机抽取200条数据（不重复）
sample_df = df.sample(n=200, random_state=42)  # random_state 保证可重复性

# 保存为新的Excel文件
sample_df.to_excel('PrimeVul_test.xlsx', index=False)

print("✅ 随机抽取完成，已保存为 PrimeVul_test.xlsx")


#
# import pandas as pd
#
# # 读取原始Excel文件
# df = pd.read_excel('PrimeVul_all.xlsx')
#
# # 筛选出 search_commit_msg 不为空的行
# df_filtered = df[df['search_commit_msg'].notna() & (df['search_commit_msg'].str.strip() != '')]
#
# # 检查是否有足够的数据
# if len(df_filtered) < 200:
#     print(f"⚠️ 警告：只有 {len(df_filtered)} 条数据的 'search_commit_msg' 不为空，少于 200 条。")
# else:
#     # 随机抽取200条数据（不重复）
#     sample_df = df_filtered.sample(n=200, random_state=42)  # random_state 保证可重复性
#
#     # 保存为新的Excel文件
#     sample_df.to_excel('PrimeVul_test.xlsx', index=False)
#
#     print("✅ 随机抽取完成，已保存为 PrimeVul_test.xlsx")