import pandas as pd
import ast

# 读取 Excel 文件
df = pd.read_excel('MegaVul_test.xlsx')

# 检查是否存在 'cwe_ids' 列
if 'cwe_ids' not in df.columns:
    print("文件中不存在 'cwe_ids' 列")
else:
    # 将空值替换为 'CWE-Other'
    df['cwe_ids'] = df['cwe_ids'].fillna('CWE-Other')

    # 定义函数：提取第一个 CWE ID
    def extract_first_cwe(cwe_str):
        try:
            # 尝试将字符串转换为列表
            cwe_list = ast.literal_eval(cwe_str)
            if isinstance(cwe_list, list) and len(cwe_list) > 0:
                return cwe_list[0]
            else:
                return cwe_str  # 如果不是列表，返回原值
        except Exception:
            return cwe_str  # 如果解析失败，返回原值

    # 应用函数，提取第一个 CWE ID
    df['cwe_ids'] = df['cwe_ids'].apply(extract_first_cwe)
    unique_cwe_ids = df['cwe_ids'].dropna().unique()
    print(f"共有 {len(unique_cwe_ids)} 种不同的 CWE ID")
    
    # 统计每种 CWE ID 的数量
    cwe_counts = df['cwe_ids'].value_counts()

    # 输出结果
    print("每种 CWE ID 的数量：")
    print(cwe_counts)