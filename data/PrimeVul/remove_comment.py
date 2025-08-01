import pandas as pd
import re


def remove_comments_and_empty_lines(code):
    # 删除单行和多行注释
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    # 删除空行
    code = re.sub(r'^\s*\n', '', code, flags=re.MULTILINE)

    return code.strip()


# 读取CSV文件
df = pd.read_csv('PrimeVul_new.csv')

# 处理func_before列中的每个代码字符串
df['func_before_recom'] = df['func'].apply(remove_comments_and_empty_lines)

# 将处理后的结果与原始数据一起存储到新的CSV文件
df.to_csv('PrimeVul_new_recom.csv', index=False)

print("处理完成，结果已保存到 PrimeVul_new_recom.csv")