import numpy as np
score=np.array([[80,90,85],[70,85,92],[88,78,90],[97,67,87]])
name=["张三","李四","王五","赵六"]
course=["语文","数学","英语"]
each_one_sum=score.sum(axis=1)
each_one_mean=score.mean(axis=1)
all_sum=score.sum(axis=0)
all_mean=score.mean(axis=0)
all_sum_mean=score.mean()
for i,n in enumerate(name):
    print(f"{n}的总分为{each_one_sum[i]}平均分为{each_one_mean[i]}")

for i,c in enumerate(course):
    print(f"{c}课程的总分为{all_sum[i]}平均分为{all_mean[i]}")
    
print(f"全班总平均分为{all_sum_mean}")