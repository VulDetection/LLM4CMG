import pandas as pd
model = "deepseekv3"
# 读取数据
df_test = pd.read_excel('bigvul_test.xlsx')
df_codellama = pd.read_excel('bigvul_{}.xlsx'.format(model))
df_evaluation = pd.read_excel('bigvul_evaluation_llm.xlsx')

# 处理CWE ID列的空值
df_test['CWE ID'] = df_test['CWE ID'].fillna('CWE-Other')

# 选取指定的CWE类型
selected_cwes = ['CWE-119', 'CWE-125', 'CWE-20']
mask = df_test['CWE ID'].isin(selected_cwes)

# 筛选并提取指定列
test_cols = ['CWE ID', 'Summary.1', 'Background', 'Impact', 'Fix', 'func_before_recom', 'patch']
codellama_cols = ['Summary', 'Background', 'Impact', 'Fix']
evaluation_cols = ['bigvul_{}_gpt'.format(model),'bigvul_{}_deepseekv3'.format(model),'bigvul_{}_gemini'.format(model),'bigvul_{}_glm4'.format(model), 'bigvul_{}_human'.format(model)]

df_test_filtered = df_test.loc[mask, test_cols].copy()
df_codellama_filtered = df_codellama.loc[mask, codellama_cols].copy()
df_evaluation_filtered = df_evaluation.loc[mask, evaluation_cols].copy()

# 重命名列
df_test_filtered = df_test_filtered.rename(columns={
    'Summary.1': 'Summary_gt',
    'Background': 'Background_gt',
    'Impact': 'Impact_gt',
    'Fix': 'Fix_gt'
})

df_codellama_filtered = df_codellama_filtered.rename(columns={
    'Summary': 'Summary_pred',
    'Background': 'Background_pred',
    'Impact': 'Impact_pred',
    'Fix': 'Fix_pred'
})

# 按CWE类型分别保存
for cwe in selected_cwes:
    mask = df_test_filtered['CWE ID'] == cwe

    # 合并三个数据框的对应行
    combined = pd.concat([
        df_test_filtered[mask].reset_index(drop=True),
        df_codellama_filtered[mask].reset_index(drop=True),
        df_evaluation_filtered[mask].reset_index(drop=True)
    ], axis=1)

    # 保存到Excel文件
    filename = '{}_{}.xlsx'.format(cwe, model)
    combined.to_excel(filename, index=False)
    print(f"已保存 {cwe} 的数据到 {filename}")