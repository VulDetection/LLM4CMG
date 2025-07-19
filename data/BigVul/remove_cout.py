import pandas as pd
import re


def remove_debug_and_output_code(code):
    # 定义调试和输出相关的关键词
    debug_keywords = ["printf", "fprintf", "debug", "log", "trace"]
    patterns = [r'\b' + keyword + r'\b' for keyword in debug_keywords]

    # 删除包含调试和输出关键词的代码行
    for pattern in patterns:
        code = re.sub(pattern, '', code, flags=re.IGNORECASE)

    # 删除空行
    code = re.sub(r'^\s*\n', '', code, flags=re.MULTILINE)

    return code.strip()


# 读取CSV文件
df = pd.read_csv('BigVul_new_recom.csv')

# 处理func_before列中的每个代码字符串
df['func_before_cleaned'] = df['func_before_recom'].apply(remove_debug_and_output_code)

# 将处理后的结果与原始数据一起存储到新的CSV文件
df.to_csv('BigVul_new_recom_cleaned.csv', index=False)

print("处理完成，结果已保存到 BigVul_new_recom_cleaned.csv")