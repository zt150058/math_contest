import pandas as pd

medal_counts = pd.read_csv('F:/美赛/2025_MCM-ICM_Problems/2025_MCM-ICM_Problems/2025_Problem_C_Data/2025_Problem_C_Data/summerOly_medal_counts.csv', encoding='latin1')
iso_df=pd.read_excel('F:/美赛/2025_MCM-ICM_Problems/2025_MCM-ICM_Problems/2025_Problem_C_Data/2025_Problem_C_Data/ISO.xlsx"')#以上两个换成本地地址

medal_counts['NOC'] = medal_counts['NOC'].str.strip().str.upper()
iso_df['country'] = iso_df['country'].str.strip().str.upper()

merge_df=pd.merge(medal_counts,iso_df[['country','NOC']],left_on='NOC',right_on='country',how='left')

unmatched=merge_df[~merge_df['NOC'].isin(iso_df['country'])]
if not unmatched.empty:
    print("以下 NOC 在 ISO 表中找不到对应的国家代码：")
    print(unmatched)


merge_df['NOC']=merge_df['NOC_y']
merge_df=merge_df.drop(columns=['NOC_x','NOC_y','country'],errors='ignore')

merge_df['NOC']=merge_df['NOC'].fillna('Unknown')

# 删除 'NOC' 列中值为 'Unknown' 的行
merge_df = merge_df[merge_df['NOC'] != 'Unknown']

# 保存合并后的文件summerOly_medal_counts_replaced.csv,换成你的本地地址
merge_df.to_csv('F:/美赛/2025_MCM-ICM_Problems/2025_MCM-ICM_Problems/2025_Problem_C_Data/2025_Problem_C_Data/processed/summerOly_medal_counts_replaced.csv', index=False, encoding='utf-8')

print(merge_df.head())
