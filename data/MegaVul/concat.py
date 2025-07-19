import pandas as pd
import re

# 定义清洗非法字符的函数
def remove_illegal_chars(val):
    if isinstance(val, str):
        # 去除Excel不支持的字符（控制字符）
        return re.sub(r'[\x00-\x1F\x7F]', '', val)
    return val


csv_path = 'MegaVul_new_recom.csv'
df_csv = pd.read_csv(csv_path)


excel_path = 'MegaVul_new_com_msg.xlsx'
df_excel = pd.read_excel(excel_path, usecols=['Summary', 'Background', 'Impact', 'Fix'])

# 确保两个文件的行数一致
if len(df_csv) != len(df_excel):
    print("⚠️ 警告：两个文件的行数不一致，可能会导致数据错位！")

# 合并两个DataFrame（按列拼接）
df_merged = pd.concat([df_csv, df_excel], axis=1)

# 清洗所有字符串列中的非法字符
df_merged = df_merged.applymap(remove_illegal_chars)

# 保存为新的Excel文件
output_path = 'MegaVul_all.xlsx'
df_merged.to_excel(output_path, index=False)

print(f"✅ 合并完成，结果已保存为：{output_path}")