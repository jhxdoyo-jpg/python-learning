import requests
params={'name':input("请输入英文名")}
try:
    r=requests.get('https://api.agify.io',params=params)
    r.raise_for_status()
    data=r.json()
    if data["age"]!=None:
        print(f"这个英文名{data["name"]}大概是{data["age"]}岁的人用")
    else:
        print("猜不出来喔")
except Exception as e:
    print("出错了",e)              
