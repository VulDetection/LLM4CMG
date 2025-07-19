import pandas as pd
from pathlib import Path

cwe = '20'
model = 'glm4'
# 1. 路径与对应前缀
paths = {
    'bigvul':  Path(f'BigVul/CWE-{cwe}_{model}.xlsx'),
    'megavul': Path(f'MegaVul/CWE-{cwe}_{model}.xlsx'),
    'primevul':Path(f'PrimeVul/CWE-{cwe}_{model}.xlsx')
}

# 2. 读入 + 重命名最后两列
df_list = []
for prefix, p in paths.items():
    df = pd.read_excel(p)
    # 取最后两列名
    old_cols = df.columns[-5:]
    new_cols = [f'score_gpt', 'score_deepseekv3', 'score_gemini', 'score_glm4', f'score_human']
    rename_map = dict(zip(old_cols, new_cols))
    df = df.rename(columns=rename_map)
    df_list.append(df)

# 3. 合并并保存
df_all = pd.concat(df_list, axis=0, ignore_index=True)
out_file = f'CWE-{cwe}_{model}.xlsx'
df_all.to_excel(out_file, index=False)
print(f'已生成 {out_file}')