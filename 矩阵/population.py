import pandas as pd 
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']#确保中文字体不是框框
df=pd.read_csv("population.csv",encoding="utf-8")
print(df.head())
print(df.info())
print(df.describe())
print(df[(df["Country Name"] =="China" ) & (df["Year"]==2024)])
print(df[(df["Value"]>1e8)&(df["Year"]==2024)])

countries = df[df["Country Code"].str.len() == 3]#过滤地区脏数据
total_df=countries.groupby("Year")["Value"].sum().reset_index()#重设下标使得能作图识别x轴

total_df.columns=["Year","Total"]
print(total_df)
total_df.plot(x="Year",y="Total",kind="line",marker="o")
plt.title("全世界人口趋势")
plt.xlabel("年份")
plt.ylabel("全球总人口")
plt.show()

