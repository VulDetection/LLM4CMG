import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('BigVul_test.xlsx')

# 检查是否存在 'CWE ID' 列
if 'CWE ID' not in df.columns:
    print("文件中不存在 'CWE ID' 列")
else:
    # 将空值替换为 'CWE-Other'
    df['CWE ID'] = df['CWE ID'].fillna('CWE-Other')
    unique_cwe_ids = df['CWE ID'].dropna().unique()
    print(f"共有 {len(unique_cwe_ids)} 种不同的 CWE ID")

    # 统计每种 CWE ID 的数量
    cwe_counts = df['CWE ID'].value_counts()


    # 输出结果
    print("每种 CWE ID 的数量：")
    print(cwe_counts)