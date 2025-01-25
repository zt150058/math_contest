import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

import warnings
warnings.filterwarnings('ignore')

MIN_GROWTH_RATE =0.05#默认最小增幅
MIN_VAL =1 #当Athletes_2024或event_2024未0时设一个下限

#读取数据
medal_counts_df = pd.read_csv('F:/美赛/2025_MCM-ICM_Problems/2025_MCM-ICM_Problems/2025_Problem_C_Data/2025_Problem_C_Data\summerOly_medal_counts_cleaned.csv',encoding='latin1')
#注意这里替换为你本地的summerOly_medal_counts_cleaned.csv,不是summerOly_medal_counts.csv
athletes_df=pd.read_csv('F:/美赛/2025_MCM-ICM_Problems/2025_MCM-ICM_Problems/2025_Problem_C_Data/2025_Problem_C_Data\summerOly_athletes.csv',encoding='latin1')
#同样替换成你本地路径

medal_counts_df['Year']=medal_counts_df['Year'].astype(int)
athletes_df['Year']=athletes_df['Year'].astype(int)

athletes_count=athletes_df.groupby (['NOC','Year']).size().reset_index(name='Athletes')
merged_df=pd.merge(medal_counts_df,athletes_count,on=['NOC','Year'],how='left')
events_count=athletes_df.groupby (['NOC','Year'])['Sport'].nunique().reset_index(name='Events')
merged_df=pd.merge(merged_df,events_count,on=['NOC','Year'],how='left')

#2024年数据
athletes_2024=merged_df[merged_df['Year']==2024].groupby(['NOC'])['Athletes'].mean()
events_2024=merged_df[merged_df['Year']==2024].groupby(['NOC'])['Events'].mean()

athletes_2024=athletes_2024[athletes_2024.index != 'Mixed team']
events_2024 =events_2024[events_2024.index != 'Mixed team']

#如果2024数据是0或NaN. 手动设置成MIN_VAL
athletes_2024.fillna(MIN_VAL,inplace=True)
athletes_2024[athletes_2024<MIN_VAL]= MIN_VAL
events_2024.fillna(MIN_VAL,inplace=True)
events_2024[events_2024<MIN_VAL]= MIN_VAL


def calculate_growth_rate(df,column):
    # 分组后，如果只有一条记录或首条=0，返回MIN_GROWTH_RATE
    def _growth(group):
        if len(group)>1 and group[column].iloc[0]!=0:
            ratio=(group[column].iloc[-1]/group[column].iloc[0]) ** (1/(len(group)-1))-1
            return ratio if ratio >0 else MIN_GROWTH_RATE
        else:
            return MIN_GROWTH_RATE

    result =df.groupby(['NOC']).apply(_growth).reset_index(name=f'{column}_growth')
    return result

#计算增长率
athletes_growth=calculate_growth_rate(merged_df,'Athletes')
events_growth=calculate_growth_rate(merged_df,'Events')

athletes_2024_growth=athletes_growth.set_index('NOC')['Athletes_growth_rate']
events_2024_growth=events_growth.set_index('NOC')['Events_growth_rate']

#预测2028数量  2024*（增长率+1）
athletes_2028=athletes_2024*(1+athletes_2024_growth)
events_2028=events_2024*(1+events_2024_growth)

prediction_gold={}
prediction_total={}
prediction_interval_gold={}
prediction_interval_total={}
model_gold={}

for country in merged_df['NOC'].unique():
    country_data=merged_df[merged_df['NOC']==country]
    if len(country_data)<2:
        print (f"Skipping {country},insufficient data.")
        continue

    x=country_data[['Year','Athletes','Events']]
    y_gold=country_data['Gold']
    y_total=country_data['Total']
    x_train,x_test,y_train_gold,y_test_gold =train_test_split(x,y_gold,test_size=0.2,random_state=42)
    x_train_total,x_test_total,y_train_total,y_test_total =train_test_split(x,y_total,test_size=0.2,random_state=42)


    model_gold=RandomForestRegressor(n_estimators=300,random_state=42)
    model_total=RandomForestRegressor(n_estimators=300,random_state=42)
    model_gold.fit(x_train,y_train_gold)
    model_total.fit(x_train_total,y_train_total)

    mes_gold=mean_squared_error(y_test_gold, model_gold.predict(x_test))
    mes_total=mean_squared_error(y_test_total, model_total.predict(x_test_total))

    print(f"{country}: Gold MSE={mes_gold},Total MSE={mes_total}")

    if country in athletes_2028.index and country in events_2028.index:
        x_2028=pd.DataFrame({
            'Year':[2028],
            'Athletes':[athletes_2028[country]],
            'Events':[events_2028[country]],
        })
        x_2028=x_2028.replace([np.inf,-np.inf],0).fillna(0)

        pred_gold_2028=model_gold.predict(x_2028)
        pred_total_2028=model_total.predict(x_2028)

        #不确定性区间
        interval_g=np.std([model_gold.predict(x_2028) for _ in range(100)])
        interval_t=np.std([model_total.predict(x_2028) for _ in range(100)])

        prediction_gold[country]=pred_gold_2028[0]
        prediction_total[country]=pred_total_2028[0]
        prediction_interval_gold[country]=interval_g
        prediction_interval_total[country]=interval_t

#2024未得牌国家
no_medals_2024=merged_df[
    (merged_df['Year']==2024)
    &(merged_df['Gold']==0)
    &(merged_df['Silver']==0)
    &(merged_df['Bronze']==0)
    ]

prediction_first_medal={}
for country in no_medals_2024['NOC'].unique():
    # 如果这个国家没有计算出增长率或不在index中，跳过
    if country not in athletes_2028.index or country not in events_2028.index:
        prediction_first_medal[country]=0
        continue
    #构建2028
    x_2028 = pd.DataFrame({
        'Year': [2028],
        'Athletes': [athletes_2028[country]],
        'Events': [events_2028[country]],
    })
    x_2028 = x_2028.replace([np.inf, -np.inf], 0).fillna(0)
    pred_g=model_gold.predict(x_2028)[0]
    prediction_first_medal[country]=1 if pred_g>0 else 0

count_first_medal=sum(prediction_first_medal.values())
total_first_medal=len(prediction_first_medal)
probability=count_first_medal/total_first_medal if total_first_medal>0 else 0

print(f"{count_first_medal} countries might get first medal in 2028.")
print(f"Probability={probability*100:.2f}%")

first_medal_df=pd.DataFrame({
    'NOC':list(prediction_first_medal.keys()),
    'Prediction First Medal in 2028':list(prediction_first_medal.values())
})
first_medal_df.to_csv('F:/美赛/2025_MCM-ICM_Problems/2025_MCM-ICM_Problems/model/first_medal_predictions.csv',index=False)#换成本地希望保留的位置

prediction_df=pd.DataFrame({
    'NOC':list(prediction_gold.keys()),
    'Prediction Gold Medal for 2028':list(prediction_gold.values()),
    'Prediction Total Medal for 2028':list(prediction_total.values()),
    'Gold Prediction Interval':list(prediction_interval_gold.values()),
    'Total Prediction Interval':list(prediction_interval_total.values())
})
prediction_df.to_csv('F:/美赛/2025_MCM-ICM_Problems/2025_MCM-ICM_Problems/model/2028_predictions_by_country.csv',index=False)#换成本地希望保留的位置
