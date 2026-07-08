from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese")
result = classifier("这个电影太好看了")
print(result)