import pandas as pd

#读取 iso.py中生成的processed/summerOly_medal_counts_replaced.csv
df=pd.read_csv('F:/美赛/2025_MCM-ICM_Problems/2025_MCM-ICM_Problems/2025_Problem_C_Data/2025_Problem_C_Data/processed/summerOly_medal_counts_replaced.csv', encoding='latin1')

#删除NOC未unknown的列
df_cleaned =df[df['NOC']!='Unknown']

#查看结果
print(df_cleaned.head())

#生成summerOly_medal_counts_cleaned.csv
df_cleaned.to_csv('F:/美赛/2025_MCM-ICM_Problems/2025_MCM-ICM_Problems/2025_Problem_C_Data/2025_Problem_C_Data/processed/summerOly_medal_counts_cleaned.csv',index=False)
