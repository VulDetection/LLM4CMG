# import requests
# from bs4 import BeautifulSoup
#
# # 目标网页的URL
# url = "https://github.com/ImageMagick/ImageMagick6/commit/dc070da861a015d3c97488fdcca6063b44d47a7b"
#
# # 发送HTTP请求获取网页内容
# response = requests.get(url)
#
# # 检查请求是否成功
# if response.status_code == 200:
#     # 获取HTML内容
#     html_content = response.text
#
#     # 使用BeautifulSoup解析HTML
#     soup = BeautifulSoup(html_content, 'html.parser')
#
#     # 查找包含差异的表格
#     diff_table = soup.find('table', class_='tab-size width-full DiffLines-module__tableLayoutFixed--ZmaVx')
#
#     # 如果找到表格，提取并输出内容
#     if diff_table:
#         # 提取表格的tbody部分
#         tbody = diff_table.find('tbody')
#
#         # 初始化变量来存储输出内容
#         output_content = ""
#         in_diff_block = False
#
#         # 遍历tbody中的每一行
#         for row in tbody.find_all('tr', recursive=False):
#             # 提取行中的所有单元格
#             cells = row.find_all('td')
#             # 检查是否是标记块或者变更代码块
#             if len(cells) > 2:
#                 # 包含变更标记和代码的单元格
#                 for cell in cells[2:]:
#                     if cell.has_attr('data-line-anchor'):
#                         # 包含变更标记(+或-)
#                         output_content += cell.get_text(strip=True) + "\n"
#                         in_diff_block = True
#                     else:
#                         # 包含代码变更的单元格
#                         output_content += cell.get_text(strip=True) + "\n"
#             elif len(cells) == 1 and '@@' in cells[0].get_text(strip=True):
#                 # @@ 标记块
#                 output_content += cells[0].get_text(strip=True) + "\n"
#                 in_diff_block = True
#
#         # 输出所有收集到的代码变更内容
#         print(output_content)
#     else:
#         print("未找到包含差异的表格")
# else:
#     print(f"请求失败，状态码：{response.status_code}")


import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

# 读取CSV文件
input_csv_file = 'PrimeVul_new_recom.csv'

# 读取原始数据
df = pd.read_csv(input_csv_file)


# 定义一个函数来提取commit信息
def extract_commit_info(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            diff_table = soup.find('table', class_='tab-size width-full DiffLines-module__tableLayoutFixed--ZmaVx')
            output_content = ""
            if diff_table:
                tbody = diff_table.find('tbody')
                for row in tbody.find_all('tr', recursive=False):
                    cells = row.find_all('td')
                    if len(cells) > 2:
                        for cell in cells[2:]:
                            output_content += cell.get_text(strip=True) + "\n"
                    elif len(cells) == 1 and '@@' in cells[0].get_text(strip=True):
                        output_content += cells[0].get_text(strip=True) + "\n"

                    print(output_content)
            return output_content.strip()
        else:
            return ""
    except Exception as e:
        print(f"处理URL {url} 时发生错误：{e}")
        return ""


# 初始化计数器
count = 0

# 创建一个新的DataFrame来存储结果
output_df = pd.DataFrame()

# 应用函数提取信息并立即保存到新的CSV文件
for index, row in df.iterrows():
    url = row['commit_url']
    output_content = extract_commit_info(url)
    count += 1
    print(f"处理完成第 {count} 条数据，URL: {url}")

    # 创建一个临时DataFrame
    temp_df = pd.DataFrame({'commit_url': [url], 'search_commit_msg': [output_content]})
    # 将临时DataFrame追加到输出DataFrame
    output_df = pd.concat([output_df, temp_df], ignore_index=True)
    # 将输出DataFrame保存到CSV文件
    output_df.to_csv('PrimeVul_new_recom_with_search_commit_msg.csv', index=False)

print("所有数据处理完毕并保存到新的CSV文件。")