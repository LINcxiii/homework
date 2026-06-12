# 花卉特征分类系统 
import random
import json
import time

# 数据集类
class FlowerDataset:
    def __init__(self):
        self.label_map = {0:"雏菊",1:"玫瑰",2:"郁金香",3:"向日葵",4:"莲花"}
        self.data = []
        self._create_data()

    def _create_data(self):
        for label in range(5):
            for _ in range(40):
                feature = [round(random.uniform(0,1),3) for _ in range(6)]
                self.data.append((feature, label))

    def get_total_data(self):
        return self.data

# 分类模型类
class ClassifyModel:
    def __init__(self):
        self.weight = None
        self.trained_flag = False

    def train(self,train_data):
        weight_dict = {}
        for feat,lab in train_data:
            key = sum(feat)
            weight_dict[key] = lab
        self.weight = weight_dict
        self.trained_flag = True
        print("模型训练完成，参数已收敛")

    def infer(self,feature):
        if not self.trained_flag:
            return "模型尚未完成训练，请先执行训练操作"
        s = sum(feature)
        min_gap = 999
        res_label = 0
        for total_sum,lab in self.weight.items():
            gap = abs(s - total_sum)
            if gap < min_gap:
                min_gap = gap
                res_label = lab
        return res_label

    def save_model(self,save_path="model.json"):
        with open(save_path,"w",encoding="utf-8") as f:
            json.dump(self.weight,f)
        return True

    def load_model(self,load_path="model.json"):
        try:
            with open(load_path,"r",encoding="utf-8") as f:
                self.weight = json.load(f)
            self.trained_flag = True
            return True
        except Exception:
            return False

dataset = FlowerDataset()
model = ClassifyModel()
infer_log = []

def run_train():
    train_data = dataset.get_total_data()
    model.train(train_data)

def flower_infer():
    print("请输入6个0~1之间的特征数值，空格隔开：")
    input_str = input("特征输入：")
    feat_list = list(map(float,input_str.strip().split()))
    lab = model.infer(feat_list)
    cls_name = dataset.label_map[lab]
    print(f"分类结果：该样本所属品类为【{cls_name}】\n")
    infer_log.append({"feat":feat_list,"result":cls_name,"time":time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())})

def show_all_log():
    print("\n==========历史推理记录==========")
    if len(infer_log)==0:
        print("暂无推理记录")
        return
    for idx,item in enumerate(infer_log,1):
        print(f"{idx} 记录时间:{item['time']} 分类结果:{item['result']}")
    print("================================\n")

def save_log_to_file():
    with open("infer_log.txt","w",encoding="utf-8") as f:
        for item in infer_log:
            f.write(f"{item['time']} 分类结果:{item['result']} 特征:{item['feat']}\n")
    print("全部记录已保存至 infer_log.txt\n")

def main():
    print("====花卉特征分类系统====")
    print("可用指令说明：")
    print("训练模型  ：完成模型整体训练")
    print("样本推理  ：输入特征完成品类分类")
    print("保存模型  ：将训练参数保存为model.json")
    print("加载模型  ：读取本地已保存的模型文件")
    print("查看记录  ：浏览全部历史推理数据")
    print("保存记录  ：将记录导出至本地文本文件")
    print("退出程序  ：结束程序运行\n")
    while True:
        cmd = input("请输入指令：")
        if cmd == "退出程序":
            print("程序运行结束")
            break
        elif cmd == "训练模型":
            run_train()
        elif cmd == "样本推理":
            flower_infer()
        elif cmd == "保存模型":
            model.save_model()
            print("模型文件已保存为 model.json\n")
        elif cmd == "加载模型":
            flag = model.load_model()
            if flag:
                print("本地模型加载完成\n")
            else:
                print("未检索到模型文件，请先完成模型训练\n")
        elif cmd == "查看记录":
            show_all_log()
        elif cmd == "保存记录":
            save_log_to_file()
        else:
            print("输入指令无效，请重新输入\n")

if __name__ == "__main__":
    main()