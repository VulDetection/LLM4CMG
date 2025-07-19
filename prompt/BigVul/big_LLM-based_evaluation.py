from openai import OpenAI
import pandas as pd
from openpyxl import Workbook
from openpyxl import load_workbook
import os

# from tensorflow.python.keras.saving.saved_model.load import models_lib

# 读取数据
df_test = pd.read_excel('D:/sh_2024_2028/研究内容-4/实验/our/data/BigVul/BigVul_test.xlsx')

prompt = 'zero'


model = 'gpt'

df_pred = pd.read_excel('bigvul_{}_{}.xlsx'.format(model, prompt))

col = 'bigvul_{}_{}'.format(model, prompt)

# 创建或加载 evaluation_llm.xlsx

output_file = 'bigvul_evaluation_llm.xlsx'

# 设置OpenAI客户端
client = OpenAI(
    base_url='https:/',
    api_key='sk-',
)

# 定义一个函数来分析代码并返回结果
def analyze_code(code, commit_code, commit_msg):
    code = code[:300]
    commit_code = commit_code[:50]
    commit_msg = commit_msg[:50]
    messages = [
        {
            "role": "user",
            "content": "Here is a piece of software bug code, the changes made in the corresponding commit and the corresponding commit message."
                       "Please rate each commit message on a scale of 1 to 5, where higher is better."
                       "There are the following indicators: 1. Accurately summarize and submit change information; 2. Natural and concise statements; 3. Help software engineers quickly understand change information."
                       "Expected output format:"
                       "Score: <your rating>"
                       "Input:"
                       "Code: {}"
                       "Code Change: {}"
                       "Score: {}".format( code, commit_code, commit_msg),
        }
    ]

    response = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
        # model="deepseek-chat",
        # model="gemini-2.0-flash",
        # model="gemini-2.0-flash-lite-preview-02-05",
        # model="gpt-4-turbo",
        temperature=0.1,
        top_p=1,
    )
    for message in response.choices:
        print(message.message.content)
    result = {}
    for message in response.choices:
        for line in message.message.content.split('\n'):
            if line.startswith('Score:'):
                result['Score'] = line[len('Score: '):].strip()
    return result



if not os.path.exists(output_file):
    df_eval = pd.DataFrame()
else:
    df_eval = pd.read_excel(output_file)

# 添加新列（如果不存在）
if col not in df_eval.columns:
    df_eval[col] = None

# 确保 evaluation 文件行数与测试集一致
if len(df_eval) != len(df_test):
    df_eval = df_eval.reindex(range(len(df_test)))

# 遍历并填充评分
for i in range(len(df_test)):
    code = df_test.loc[i, 'func_before_recom']
    patch = df_test.loc[i, 'patch']
    summary = df_pred.loc[i, 'Summary']
    background = df_pred.loc[i, 'Background']
    impact = df_pred.loc[i, 'Impact']
    fix = df_pred.loc[i, 'Fix']
    commit_msg = f"Summary: {summary}\nBackground: {background}\nImpact: {impact}\nFix: {fix}"

    result = analyze_code(code, patch, commit_msg)
    score_str = result.get('Score', '')
    if score_str.isdigit():
        score = score_str
    else:
        score = 3  # 或者设为 0，取决于你的需求
    print("分数:", score)
    df_eval.at[i, col] = score

    print(f"处理完成第 {i+1} 个")

# 保存到 evaluation_llm.xlsx
df_eval.to_excel(output_file, index=False)
print(f"✅ 所有评分已保存到 {output_file} 的 bigvul_gpt_gpt 列中")




# import pandas as pd
# # 读取 Excel 文件
# df = pd.read_excel('evaluation_llm.xlsx')
# # 计算 bigvul_gpt_gpt 列的均值（自动忽略非数字和空值）
# mean_score = df['bigvul_gpt_gpt'].mean()
# print(f"bigvul_gpt_gpt 列的均值为：{mean_score:.4f}")
#
#
# import pandas as pd
# # 读取 Excel 文件
# df = pd.read_excel('evaluation_llm.xlsx')
#
# # 计算所有数值列的均值（自动忽略非数字和空值）
# mean_scores = df.mean(numeric_only=True)
#
# # 打印结果
# print("各列的均值如下：")
# print(mean_scores)
