import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

# ==================== 1. 造医学数据集 ====================
# 三类：呼吸内科(0)、心内科(1)、消化内科(2)，每类 50 条 → 共 150 条
data = [
    # ── 呼吸内科 ──
    {"text": "患者反复咳嗽咳痰三周，伴有胸闷气短", "label": 0},
    {"text": "发热三天体温39度，咽喉红肿有脓点", "label": 0},
    {"text": "胸片显示双肺纹理增粗紊乱模糊", "label": 0},
    {"text": "夜间阵发性呼吸困难不能平卧入睡", "label": 0},
    {"text": "咳黄痰伴有发热畏寒全身肌肉酸痛", "label": 0},
    {"text": "支气管镜检见大量脓性分泌物潴留", "label": 0},
    {"text": "听诊双肺可闻及广泛哮鸣音干啰音", "label": 0},
    {"text": "血常规提示白细胞计数升高至一万五", "label": 0},
    {"text": "患者有慢性阻塞性肺疾病病史八年整", "label": 0},
    {"text": "感冒后咳嗽持续两月未见好转迹象", "label": 0},
    {"text": "核酸检测阳性伴有双肺磨玻璃样改变", "label": 0},
    {"text": "午后低热盗汗乏力消瘦疑似肺结核活动期", "label": 0},
    {"text": "喘息发作每年超过三次肺功能提示阻塞", "label": 0},
    {"text": "患者吸烟史三十年近期咳血丝痰带血", "label": 0},
    {"text": "过敏性鼻炎十年反复打喷嚏流清涕鼻塞", "label": 0},
    {"text": "剧烈咳嗽后出现胸痛呼吸时加剧难忍", "label": 0},
    {"text": "胸部增强CT显示右肺上叶占位性病变", "label": 0},
    {"text": "长期粉尘接触史近半年咳嗽逐渐加重", "label": 0},
    {"text": "气管插管拔除后出现喉头水肿呼吸困难", "label": 0},
    {"text": "睡眠呼吸暂停综合征夜间血氧最低降至75%", "label": 0},
    {"text": "儿童反复喘息确诊为支气管哮喘急性发作", "label": 0},
    {"text": "间质性肺病患者活动后喘憋血氧下降明显", "label": 0},
    {"text": "胸腔积液穿刺抽出黄色浑浊液体约六百毫升", "label": 0},
    {"text": "感冒引起急性鼻窦炎头痛面部压痛明显", "label": 0},
    {"text": "肺癌术后复诊CT未见明显复发转移征象", "label": 0},
    {"text": "患者有哮喘病史感冒诱发严重呼吸困难", "label": 0},
    {"text": "慢性咳嗽两个月夜间加重大吠样干咳", "label": 0},
    {"text": "支原体肺炎流行季发热咳嗽伴皮疹出现", "label": 0},
    {"text": "气胸患者突发胸痛呼吸困难叩诊鼓音明显", "label": 0},
    {"text": "慢性咽炎反复发作咽部异物感清嗓频繁", "label": 0},
    {"text": "矽肺患者肺功能进行性下降活动受限严重", "label": 0},
    {"text": "刺激性干咳伴低热疑似早期肺癌需排查", "label": 0},
    {"text": "哮喘持续状态经支气管扩张剂治疗后缓解", "label": 0},
    {"text": "流感季节高热全身酸痛咽痛咳嗽一周余", "label": 0},
    {"text": "肺栓塞患者突发胸痛咯血呼吸困难三联征", "label": 0},
    {"text": "纤维支气管镜检查见左主支气管新生物堵塞", "label": 0},
    {"text": "支气管扩张症患者大量咳脓痰晨起时更甚", "label": 0},
    {"text": "呼吸衰竭患者动脉血气分析二氧化碳分压升高", "label": 0},
    {"text": "过敏体质春季花粉诱发打喷嚏流涕眼痒不适", "label": 0},
    {"text": "患者畏寒高热咳嗽胸片提示大叶性肺炎改变", "label": 0},
    {"text": "过敏性哮喘患者接触尘螨后立即喘息发作", "label": 0},
    {"text": "上呼吸道感染后出现急性支气管炎剧烈咳嗽", "label": 0},
    {"text": "胸膜间皮瘤患者持续性钝痛呼吸困难体重下降", "label": 0},
    {"text": "肺脓肿患者高热咳大量脓臭痰伴杵状指出现", "label": 0},
    {"text": "双侧肺门淋巴结肿大气管镜活检提示结节病", "label": 0},
    {"text": "氧饱和度持续低于90%患者需要无创呼吸机支持", "label": 0},
    {"text": "支气管激发试验阳性确诊咳嗽变异性哮喘", "label": 0},
    {"text": "患者误吸后出现吸入性肺炎右下肺实变阴影", "label": 0},
    {"text": "花粉症季节性发作鼻塞喷嚏眼结膜充血水肿", "label": 0},
    {"text": "重症肺炎合并急性呼吸窘迫综合征转ICU治疗", "label": 0},

    # ── 心内科 ──
    {"text": "胸骨后压榨性疼痛持续三十分钟含服硝酸甘油不缓解", "label": 1},
    {"text": "心电图显示窦性心动过速心率一百一十次每分", "label": 1},
    {"text": "心脏彩超提示左心室射血分数降低至百分之三十五", "label": 1},
    {"text": "患者有高血压病史十年最高达180/110毫米汞柱", "label": 1},
    {"text": "心悸胸闷活动后加重爬楼梯到二楼即气喘", "label": 1},
    {"text": "动态心电图提示有频发室性早搏二联律出现", "label": 1},
    {"text": "患者突发心悸心电图示室上性心动过速持续", "label": 1},
    {"text": "急性心肌梗死入院心肌酶CK-MB显著升高为正常五倍", "label": 1},
    {"text": "双下肢对称性凹陷性水肿考虑右心功能不全所致", "label": 1},
    {"text": "冠脉造影显示左前降支中段狭窄已达百分之九十", "label": 1},
    {"text": "心尖区可闻及舒张期隆隆样杂音二尖瓣狭窄可能", "label": 1},
    {"text": "起搏器植入术后复查心电图显示起搏功能良好", "label": 1},
    {"text": "心脏MRI提示肥厚型梗阻性心肌病室间隔厚度增加", "label": 1},
    {"text": "患者长期服用华法林抗凝INR控制目标2.0-3.0之间", "label": 1},
    {"text": "感染性心内膜炎血培养三次均见金黄色葡萄球菌", "label": 1},
    {"text": "心包积液大量心影向两侧扩大呈烧瓶状改变", "label": 1},
    {"text": "射频消融术后复查心电图未见预激波出现及复发", "label": 1},
    {"text": "胸主动脉瘤支架植入术后CTA复查位置良好无内漏", "label": 1},
    {"text": "高脂血症总胆固醇显著升高达8.7毫摩尔每升以上", "label": 1},
    {"text": "患者突发晕厥心电图示三度房室传导阻滞抢救", "label": 1},
    {"text": "冠状动脉搭桥术后患者恢复良好伤口愈合无感染", "label": 1},
    {"text": "心脏瓣膜置换术后需长期抗凝治疗定期监测凝血", "label": 1},
    {"text": "急性左心衰竭端坐呼吸咳粉红色泡沫样痰抢救", "label": 1},
    {"text": "先天性心脏病房间隔缺损封堵术后随访复查正常", "label": 1},
    {"text": "阵发性房颤发作时心率极不规则心电图可确诊", "label": 1},
    {"text": "运动平板试验阳性提示存在心肌缺血需进一步检查", "label": 1},
    {"text": "心脏移植术后免疫抑制治疗密切监测排异反应发生", "label": 1},
    {"text": "主动脉瓣重度狭窄跨瓣压差超过五十毫米汞柱需手术", "label": 1},
    {"text": "室壁瘤形成于心梗后三月心脏彩超发现矛盾运动区域", "label": 1},
    {"text": "深静脉血栓形成后出现胸痛呼吸困难警惕肺栓塞", "label": 1},
    {"text": "PCI术后口服双抗治疗阿司匹林联合氯吡格雷至少一年", "label": 1},
    {"text": "风湿性心脏病联合瓣膜病变累及二尖瓣和主动脉瓣", "label": 1},
    {"text": "病态窦房结综合征动态心电图记录到长间歇大于三秒", "label": 1},
    {"text": "儿茶酚胺敏感性室速运动或情绪激动可诱发恶性心律失常", "label": 1},
    {"text": "心脏破裂是心梗后机械并发症死亡率极高需紧急手术", "label": 1},
    {"text": "限制型心肌病舒张功能严重受损心房压力显著升高腹水", "label": 1},
    {"text": "B型钠尿肽前体升高至三千pg/mL以上提示心力衰竭", "label": 1},
    {"text": "右心导管检查肺动脉压力显著升高确诊肺动脉高压", "label": 1},
    {"text": "心源性休克患者血压难以维持多巴胺剂量逐步递增效果不佳", "label": 1},
    {"text": "冠脉痉挛变异性心绞痛多在凌晨静息时发作胸痛", "label": 1},
    {"text": "卵圆孔未闭导致反常栓塞青年卒中需封堵治疗", "label": 1},
    {"text": "大量心包积液导致心脏压塞血压骤降颈静脉怒张典型", "label": 1},
    {"text": "房室折返性心动过速附加旁道参与折返环路形成", "label": 1},
    {"text": "酒精性心肌病有长期大量饮酒史戒酒后心功能有所改善", "label": 1},
    {"text": "围产期心肌病产后一月出现进行性心衰症状加重", "label": 1},
    {"text": "迷走神经张力过高导致窦性心动过缓夜间最慢三十几次", "label": 1},
    {"text": "主动脉夹层患者突发撕裂样胸背痛D-二聚体显著升高", "label": 1},
    {"text": "甲状腺功能亢进性心脏病心率快心房颤动为主要表现", "label": 1},
    {"text": "运动员心脏表现为窦性心动过缓心脏彩超正常无需过度干预", "label": 1},
    {"text": "左心耳封堵术后脑卒中高危房颤患者可减少血栓栓塞事件", "label": 1},

    # ── 消化内科 ──
    {"text": "上腹部规律性疼痛每于饭后两小时发作伴反酸", "label": 2},
    {"text": "呕血量约五百毫升色鲜红混有胃内容物需紧急止血", "label": 2},
    {"text": "大便隐血持续阳性三月近一周排柏油样黑便三次", "label": 2},
    {"text": "肝功能谷丙转氨酶高达240U/L提示肝细胞损伤", "label": 2},
    {"text": "进食油腻后右上腹疼痛向右肩背部放射超声示胆囊结石", "label": 2},
    {"text": "吞咽困难进行性加重半年从固体食物到流质均受阻", "label": 2},
    {"text": "Hp呼气试验强阳性需四联根除治疗联合两种抗生素", "label": 2},
    {"text": "腹泻水样便每日十余次伴有恶心呕吐发热38.5℃", "label": 2},
    {"text": "酒精性肝硬化失代偿期腹水大量脾功能亢进黄疸", "label": 2},
    {"text": "急性胰腺炎血淀粉酶显著升高至正常上限五倍", "label": 2},
    {"text": "溃疡性结肠炎患者腹泻黏液脓血便每日十余次腹痛", "label": 2},
    {"text": "肠易激综合征腹泻型精神紧张后腹痛欲便排便后缓解", "label": 2},
    {"text": "反流性食管炎患者胸骨后烧灼感平卧时症状加重", "label": 2},
    {"text": "肝炎后肝硬化患者需长期抗病毒治疗恩替卡韦每天一片", "label": 2},
    {"text": "克罗恩病累及回盲部肠镜见铺路石样改变纵行溃疡", "label": 2},
    {"text": "脂肪肝重度腹部超声提示肝脏回声增强肝肾功能正常", "label": 2},
    {"text": "胃镜检查发现胃角溃疡取活检病理回报待查排除恶变", "label": 2},
    {"text": "结肠多发息肉活检病理回报：管状腺瘤伴低级别上皮内瘤变", "label": 2},
    {"text": "肝硬化门脉高压食管胃底静脉曲张重度有破裂出血风险", "label": 2},
    {"text": "急性化脓性胆管炎夏科三联征高热腹痛黄疸急诊解除梗阻", "label": 2},
    {"text": "药物性肝损伤对乙酰氨基酚过量服用三天转氨酶飙升", "label": 2},
    {"text": "肠结核患者右下腹可触及包块PPD试验强阳性改变", "label": 2},
    {"text": "胃切除术后吻合口溃疡反复疼痛黑便需再次内镜干预", "label": 2},
    {"text": "重度萎缩性胃炎伴肠上皮化生需每年胃镜随访监测", "label": 2},
    {"text": "自身免疫性肝炎ANA阳性肝穿刺可见界面炎及浆细胞浸润", "label": 2},
    {"text": "缺血性肠病多见于老年人突发腹痛便血有心房颤动病史", "label": 2},
    {"text": "肝脓肿患者高热肝区叩痛超声引导下穿刺抽出脓液", "label": 2},
    {"text": "胃间质瘤内镜切除术后复查创面愈合良好切缘阴性", "label": 2},
    {"text": "贲门失弛缓症吞咽困难反食夜间呛咳食管测压可确诊", "label": 2},
    {"text": "甲型肝炎急性感染粪口传播自限性疾病注意休息饮食", "label": 2},
    {"text": "肝癌术后甲胎蛋白监测降至正常提示切缘阴性无残留", "label": 2},
    {"text": "一过性下食管括约肌松弛导致胃食管反流症状反复发作", "label": 2},
    {"text": "肠梗阻患者肛门停止排便排气腹部平片见多个液气平面", "label": 2},
    {"text": "阑尾穿孔继发弥漫性腹膜炎需急诊剖腹探查引流冲洗", "label": 2},
    {"text": "结肠癌术后需辅以化疗FOLFOX方案每两周一周期共六个周期", "label": 2},
    {"text": "暴发性肝衰竭肝性脑病凝血功能障碍需紧急肝移植评估", "label": 2},
    {"text": "十二指肠球部溃疡活动期可见溃疡底部有裸露血管残端", "label": 2},
    {"text": "麻痹性肠梗阻多见于腹部手术后肠道功能暂时恢复缓慢", "label": 2},
    {"text": "胆总管结石并发胆管炎ERCP取石加鼻胆管引流有效", "label": 2},
    {"text": "直肠中分化腺癌盆腔增强MRI未见淋巴结转移可先手术", "label": 2},
    {"text": "肠系膜上动脉栓塞患者突发剧烈腹痛有心房颤动病史", "label": 2},
    {"text": "牛奶过敏引起过敏性胃肠炎腹泻便血停用牛奶后好转", "label": 2},
    {"text": "门静脉血栓加重门脉高压脾大腹水难治需抗凝治疗", "label": 2},
    {"text": "先天性巨结肠患儿胎粪排出延迟腹胀明显灌肠后缓解", "label": 2},
    {"text": "食管异物内镜下取出术鱼骨横于食管中段取出顺利", "label": 2},
    {"text": "假性囊肿为急性胰腺炎常见并发症直径超六厘米需引流", "label": 2},
    {"text": "Barrett食管为长期反流导致食管下段鳞状上皮被柱状上皮取代", "label": 2},
    {"text": "肝豆状核变性铜代谢异常铜蓝蛋白降低角膜可见K-F环", "label": 2},
    {"text": "肝功能Child-Pugh B级肝硬化患者术后并发症风险显著升高", "label": 2},
    {"text": "胃扭转需急诊复位表现为剧烈上腹痛恶心干呕无法下胃管", "label": 2},
]

# 标签映射
label_names = {0: "呼吸内科", 1: "心内科", 2: "消化内科"}
print(f"数据集大小: {len(data)} 条")
print(f"每类: 50 条 → {label_names}")

# ==================== 2. 划分训练/测试集 ====================
train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42, stratify=[d["label"] for d in data]
)
print(f"训练集: {len(train_data)} 条 | 测试集: {len(test_data)} 条")

# ==================== 3. 加载 tokenizer 和模型 ====================
model_name = "bert-base-chinese"
num_labels = 3

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
)

# ==================== 4. Tokenize 数据 ====================
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

train_dataset = Dataset.from_list(train_data).map(tokenize_fn, batched=True)
test_dataset = Dataset.from_list(test_data).map(tokenize_fn, batched=True)

# ==================== 5. 评估原始模型（微调前基线） ====================
print("\n===== 评估原始 BERT（微调前基线）=====")
raw_trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./raw_eval",
        per_device_eval_batch_size=16,
        report_to="none",
        remove_unused_columns=False,  # 重要：保留 label 列
    ),
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=-1))},
)
raw_result = raw_trainer.evaluate()
raw_acc = raw_result["eval_accuracy"]
print(f"原始 BERT 准确率: {raw_acc:.4f}（≈{raw_acc*100:.1f}%，接近随机 33%）")

# ==================== 6. 配置 LoRA ====================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],  # 挂到注意力层的 Q 和 V
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出类似: trainable params: 294,912 || all params: 102,267,648 || trainable%: 0.2883%

# ==================== 7. 训练配置 ====================
output_dir = "./medical_lora_output"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,                    # 小数据集多跑几轮
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,                     # LoRA 学习率稍大
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),         # 有 GPU 就用混合精度
    report_to="none",
)

# ==================== 8. 开始训练 ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=-1))},
)

print("\n===== 开始 LoRA 微调 =====")
start_time = time.time()
trainer.train()
train_time = time.time() - start_time
print(f"\n训练时长: {train_time:.1f} 秒 ({train_time/60:.1f} 分钟)")

# ==================== 9. 评估微调后模型 ====================
lora_result = trainer.evaluate()
lora_acc = lora_result["eval_accuracy"]
print(f"\n原始 BERT 准确率:  {raw_acc:.4f} ({raw_acc*100:.1f}%)")
print(f"LoRA 微调后准确率: {lora_acc:.4f} ({lora_acc*100:.1f}%)")
print(f"准确率提升:        {lora_acc - raw_acc:.4f} (+{(lora_acc - raw_acc)*100:.1f} 个百分点)")

# 详细分类报告
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids
print("\n分类报告:")
print(classification_report(y_true, y_pred, target_names=list(label_names.values())))

# ==================== 10. 显存占用 ====================
if torch.cuda.is_available():
    mem_bytes = torch.cuda.max_memory_allocated()
    mem_gb = mem_bytes / (1024 ** 3)
    print(f"\n显存峰值占用: {mem_gb:.2f} GB")
else:
    print("\n未检测到 GPU，无法统计显存")

# ==================== 11. 保存 LoRA 权重 ====================
lora_save_path = "./my_medical_lora"
model.save_pretrained(lora_save_path)
print(f"\nLoRA 权重已保存到: {lora_save_path}")
print(f"(保存的只是适配器权重，约 1-3 MB，不包含整个 BERT)")

# ==================== 12. 画 Loss 曲线 ====================
logs = trainer.state.log_history
train_loss = [(log["step"], log["loss"]) for log in logs if "loss" in log]
eval_logs = [(log["epoch"], log["eval_accuracy"], log["eval_loss"])
             for log in logs if "eval_accuracy" in log]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss 曲线
steps, losses = zip(*train_loss)
ax1.plot(steps, losses, "b-", linewidth=1)
ax1.set_xlabel("Step")
ax1.set_ylabel("Loss")
ax1.set_title("训练 Loss 下降曲线")
ax1.grid(True, alpha=0.3)

# 准确率曲线
epochs, accs, eval_losses = zip(*eval_logs)
ax2.plot(epochs, [a * 100 for a in accs], "g-o", label="准确率")
ax2.axhline(y=raw_acc * 100, color="red", linestyle="--", label=f"原始 BERT ({raw_acc*100:.1f}%)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("验证集准确率变化")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lora_training_curve.png", dpi=150)
plt.show()
print("图表已保存到: lora_training_curve.png")

# ==================== 13. 汇总记录 ====================
print("\n" + "=" * 50)
print("📊 Day 23.5 四项指标汇总")
print("=" * 50)
print(f"① 训练时长:   {train_time:.1f} 秒 ({train_time/60:.1f} 分钟)")
if torch.cuda.is_available():
    print(f"② 显存占用:   {mem_gb:.2f} GB")
else:
    print(f"② 显存占用:   CPU 模式（无 GPU）")
print(f"③ Loss 曲线:  见 lora_training_curve.png")
print(f"④ 准确率提升: 原始 {raw_acc*100:.1f}% → LoRA {lora_acc*100:.1f}%（+{(lora_acc-raw_acc)*100:.1f} 个百分点）")
print("=" * 50)