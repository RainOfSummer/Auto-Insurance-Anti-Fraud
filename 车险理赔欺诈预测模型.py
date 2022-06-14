"""
项目名称：综合风险预测模型
开发时间：20220609
开发者：liwenjie
"""



# 设置路径
input_path = "Y:\PythonProgram\data\whoel_sz_f_202210_202212.csv"
dic_path = "Y:\PythonProgram\data\whoel_sz_f_202210_202212_dic.json"
# output_path = ""



# 导库
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import json
from sklearn.model_selection import train_test_split as TTS
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier as XGBC


# 导入数据
data = pd.read_csv(input_path)
# data_copy = data.copy()  # 防止数据丢失



# 构造标签  {1:is_new_inv=1， 0：is_new_inv=0}
data["Y"] = 0
data.loc[data.is_new_inv == "是", "Y"] = 1




# 特征预处理
## 编写函数
def LookFeature(column, data):
    """输出异常vs非异常的数据分布对比图"""
    ab = data.loc[data["Y"]==1, column]
    normal = data.loc[data["Y"]==0, column]
    plt.figure(figsize=(10,8), dpi=80)
    sns.kdeplot(normal, shade=True, color="#01a2d9", label="is_new_inv", alpha=.5)
    sns.kdeplot(ab, shade=True, color="#dc2624", label="not_new_inv", alpha=.9)
    plt.title("is_new_inv" + column)
    plt.show()


def encoderYN(column, data):
    """编码{Y:1, N:0, nan:-1}"""
    data[column] = data[column].map({"Y":1, "N":0, "是":1, "否":0, "1":1, "0":0, "1.0":1, "0.0":0, 1:1, 0:0})
    data[column] = data[column].fillna(-1)
    return data



# 去除、删除重单号
index_drop = data.loc[(data["report_no"].duplicated(keep=False)) & data["is_new_inv"].isnull()].index
data = data.drop(index=index_drop)
if "report_no" in data.columns:
    data = data.drop(columns=["report_no"])
data.index = range(data.shape[0])  # 恢复索引


# 时间、地点、次数相关数据处理
data["is_na_s_pigai_shijian"] = 0  # 构建是否为空的衍生特征
data.loc[data.s_pigai_shijian.isnull(), "is_na_s_pigai_shijian"] = 1
data["s_pigai_shijian"] = data["s_pigai_shijian"].fillna(-1)

data["is_na_s_endor_times"] = 0  # 构建是否为空的衍生特征
data.loc[data.s_endor_times.isnull(), "is_na_s_endor_times"] = 1
data["s_endor_times"] = data["s_endor_times"].fillna(-1)

data["is_na_car_survey_date__report_date"] = 0  # 构建是否为空的衍生特征
data.loc[data.car_survey_date__report_date.isnull(), "is_na_car_survey_date__report_date"] = 1
data["car_survey_date__report_date"] = data["car_survey_date__report_date"].fillna(-1)

data["accident_time"] = pd.to_datetime((data.accident_date)).dt.hour  #
data = data.drop(columns=["report_date", "accident_date"])    # 删除报案时间和事故时间

data.call_times = data.call_times.fillna(-1)
data.on_port_times = data.on_port_times.fillna(-1)
data.not_on_port_times = data.not_on_port_times.fillna(-1)



# 缺失值过多缺失值的特征处理
data.loc[data.is_survey_reassign.isnull(), "is_survey_reassign"] = 0

data["is_exempt_local"] = data.survey_mode.map(lambda x : 1 if x in (3, 5, 6) else 0)  # 根据查勘类型补全是否免现场

data.policy_pre_pay = data.policy_pre_pay.fillna(-1)

data = data.drop(columns=["is_third_agent_report"
                        , "is_wait_call"
                        , "flag_self_service"
                        , "drive_sex"
                        , "accident_type_detail"
                        , "accident_responsibility"
                        , "damage_climate"
                        , "dispose_style"
                        , "accident_mileage"
                        , "is_designate_accident"
                        , "risk_model_score"
                        , "special_remind"
                        , "voiceprint_blacklist_reportno"
                        , "jp_remark"])  # 缺失值过多或事后特征，删除其字段，这里可以处理得相对保守，可以根据建模规范要求的数据饱和度进行删除

# 对one_condition_flag出现的个别异常字符串进行删除处理
cat_columns = data.select_dtypes(include=["object"]).columns
if "one_condition_flag" in cat_columns:
    data = data[~(data.one_condition_flag.str.isnumeric() == False)]
    data.index = range(data.shape[0])
data.one_condition_flag = data.one_condition_flag.fillna(-1)


# 对Y_N进行编码
data.person_loss_flag = data.person_loss_flag.fillna("N")
data.is_can_driving = data.is_can_driving.fillna("Y")

columnY_N = ["is_huge_accident"
            ,"is_third_not_found"
            ,"is_violation_loading"
            ,"is_travel_across_region"
            ,"is_first_cargo_loss"
            ,"is_driver_injured"
            ,"is_panssenge_injured"
            ,"is_thi_car_loss"
            ,"is_in_thi_car_injured"
            ,"is_out_thi_car_injured"
            ,"is_in_thi_cargo_loss"
            ,"is_out_thi_cargo_loss"
            ,"is_duty_clear"
            ,"is_agent_case"
            ,"overseas_occur"
            ,"report_on_port"
            ,"person_loss_flag"
            ,"property_loss_flag"
            ,"is_can_driving"
            ,"is_whole_car_loss"
            ,"car_business_type"
            ,"is_new_power"
            ,"is_on_freeway"
            ,"is_new_inv"
            ,"is_whole_inv"]
for i in columnY_N:
    if i in data.columns:
        data = encoderYN(column=i, data=data)

# 构建是否为空的衍生特征：推修厂是否为缺失值，这个是事后特征？？？
data["is_na_repair_factory_name"] = 0  
data.loc[data.repair_factory_name.isnull(), "is_na_repair_factory_name"] = 1
data = data.drop(columns="repair_factory_name")


# 对缺失值进行-1填补
column_fill_1 = ["car_category_name"
                ,"brand_name"
                ,"manufacture_name"
                ,"series_name"
                ,"group_name"
                ,"model_name"
                ,"accident_cause_level2"
                ,"accident_cause_level3"
                ,"accident_detail"
                ,"check_department_code_name_02"
                ,"remark"
                ,"inv_pg_decrease_big_type"
                ,"inv_pg_decrease_name"]

for i in column_fill_1:
    if i in data.columns:
        data[i] = data[i].fillna(-1)


# 对少量离散型字段进行数字编码 (这里可以根据其样本占比或者标签占比对其构造衍生特征)
data.car_belong_kind_name = data.car_belong_kind_name.map({"私人":0, "企业":1, "机关":2})
data.car_use_kind_name = data.car_use_kind_name.map({"非营业":0, "营业":1})
data.mis_power_type_class = data.mis_power_type_class.map({"传统动力":0, "纯动力":1, "普通混动":2, "插电混动":3, "燃料电池":4})
data.mis_power_type_class = data.mis_power_type_class.fillna(-1)

data = data.dropna() # 删除操作之前请检查缺失值数量！！！ data.isnull().mean()
data.index = range(data.shape[0])

# 转化整数数据类型
if data.dtypes["one_condition_flag"] != "int":
    data["one_condition_flag"] = data["one_condition_flag"].astype("int")

# 到这一步缺失值已经处理完了


# 对备注文本进行特征工程（可以做正则表达式提取，继续深挖信息）
data["accident_detail_lenght"] = data["accident_detail"].astype("str").str.len()  # 提取字段长度，描述案发详细程度
if "accident_detail" in data.columns:
    data = data.drop(columns=["accident_detail"])  ###

data["accident_place_lenght"] = data["accident_place"].astype("str").str.len()  # 提取字段长度，描述案发地点详细程度
if "accident_place" in data.columns:
    data = data.drop(columns=["accident_place"])  ### 

data["remark_lenght"] = data["remark"].astype("str").str.len()  # 提取字段长度
if "remark" in data.columns:
    data = data.drop(columns=["remark"])  ###

if "survey_remark" in data.columns:
    data = data.drop(columns=["survey_remark"])  ###


# 对车的类型进行编码
data["is_top_car_category_name"] = data.car_category_name.map(lambda x: 1 if x in ("三厢轿车","运动型多功能车") else 0)
if "car_category_name" in data.columns:
    data = data.drop(columns=["car_category_name"])  ###
if "clm_veh_type_detl_name" in data.columns:
    data= data.drop(columns=["clm_veh_type_detl_name"])

data["is_top1_car_kind_name"] = data.car_kind_name.map(lambda x: 1 if x=="六座以下客车" else 0)
if "car_kind_name" in data.columns:
    data= data.drop(columns=["car_kind_name"])  ###

data["is_top15_brand_name"] = data.brand_name.map(lambda x:1 if x in ("大众", "丰田", "本田", "日产", "别克", "长安", "现代", "奥迪", "五菱", "长城", "奔驰", "宝马", "福特", "雪佛兰", "比亚迪") else 0)

data["is_top15_manufacturer_name"] = data.manufacturer_name.map(lambda x:1 if x in ("一汽大众", "上海通用", "上海大众", "东风日产", "上汽通用五菱", "一汽丰田", "吉利汽车", "长城汽车", "长安福特马自达", "广州本田", "北京现代", "广州丰田", "东风本田", "长安乘用车", "华晨宝马") else 0)

data["veh_clas_big_type_code"] = data.veh_clas_code.map(lambda x:x[0])
data["veh_clas_big_type_code"] = data.veh_clas_big_type_code.map({"A":0, "B":1, "C":2, "D":3, "E":4})
data["veh_clas_big_type_code"] = data["veh_clas_big_type_code"].astype("int")

data = data.drop(columns=["brand_name"
                    ,"series_name"
                    ,"group_name"
                    ,"model_name"
                    ,"insured_name"])  ###

# ’二级查勘机构‘和’二级承包机构是否相同
data["is_department_code_name_02_check"] = (data.check_department_code_name_02==data.department_code_name_02).astype("int")
data = data.drop(columns=["department_code_name_02"])


# 特征相关性分析（删除相关性较高的特征）
data = data.drop(columns=[#"accident_cause_02"  ### 0.989
                    'accident_cause_3'
                    #,"report_driver"  ### 报案人是否是驾驶人 1
                    ,"driver_reporter"  ### 报案人是否和驾驶人登记电话一致
                    #,"report_date__accident_date
                    ,"yanchi_hour"  ### 1
                    #,"is_na_s_pigai_shijian"  ### 1
                    ,"is_na_s_endor_times"])







# 分训练集、测试集、验证集
Y = data["Y"]
X = data.drop(columns=["report_mode"
                    ,"inv_pg_decrease_big_type"
                    ,"inv_pg_decrease_name"
                    ,"is_new_inv"
                    ,"is_whole_inv"
                    ,"policy_pre_pay"
                    ,"Y"])

Xtrain, Xtest, Ytrain, Ytest = TTS(X, Y, test_size=0.2, random_state=420)
Xtrain, Xvalid, Ytrain, Yvalid = TTS(Xtrain, Ytrain, test_size=0.2, random_state=420)

train = pd.concat([Xtrain, Ytrain], axis=1)
train.index = range(train.shape[0])
valid = pd.concat([Xvalid, Yvalid], axis=1)
valid.index = range(valid.shape[0])
test = pd.concat([Xtest, Ytest], axis=1)
test.index = range(test.shape[0])



# 多字段异常率衍生特征构造
## 编写函数
def GroupByFeature(column, newcolumn, train, valid=pd.DataFrame(), test=pd.DataFrame(), n=10):
    """
    此函数用于对多字段特征进行异常率的编码
    n为该特征字段取值的样本最低量，用于排除个别类和未出现的类别
    """

    # 计算异常率
    ratio = train.groupby([column])["Y"].mean()[train.groupby([column])["Y"].count()>n]

    # 创造字典
    df = pd.DataFrame(index=train[column].unique())
    df[newcolumn] = -1
    df.loc[ratio.index, newcolumn] = ratio

    # 映射到训练集
    train[newcolumn] = train[column].map(dict(df[newcolumn]))

    # 映射到验证集
    if valid.shape[0] != 0:
        valid[newcolumn] = valid[column].map(dict(df[newcolumn]))
        valid[newcolumn] = valid[newcolumn].fillna(-1)

    # 映射到测试集
    if test.shape[0] != 0 :
        test[newcolumn] = test[column].map(dict(df[newcolumn]))
        test[newcolumn] = test[newcolumn].fillna(-1)

    # 返回数据
    if (valid.shape[0] != 0)&(test.shape[0] != 0):
        return train, valid, test, dict(df[newcolumn])

    if (valid.shape[0] == 0)&(test.shape[0] != 0):
        return train, test, dict(df[newcolumn])

    if (valid.shape[0] != 0)&(test.shape[0] == 0):
        return train, valid, dict(df[newcolumn])

    return train, dict(df[newcolumn])


## 特征生成
cat_columns = ["check_department_code_name_02"
                ,"manufacturer_name"
                ,"veh_clas_code"
                ,"accident_cause_level1"
                ,"accident_cause_level2"
                ,"accident_cause_level3"]  # 其余数据均为数值型数据，如果不是，记得转化数据类型
dic_cat_columns = []  # 用于储存编译字典
for i in cat_columns:
    train, valid, test, locals()["dic_"+i] = GroupByFeature(column=i, newcolumn="ratio_Y_"+i, train=train, valid=valid, test=test, n=10)
    dic_cat_columns.append((i, locals()["dic_"+i]))

## 将生成的字典保存，用于跨周期数据验证及模型实际使用
try:
    with open(dic_path, "w", encoding="utf-8") as f:
        json.dump(dict(dic_cat_columns), f)
        print("特征编码字典保存成功")
except:
    print("特征编码字典保存失败")

# 删除多字段特征（也可对其编码尝试）
for i in (train, valid, test):
    i.drop(columns=cat_columns, inplace=True)
    if "policy_pre_pay" in i.columns:
        i.drop(columns=["policy_pre_pay"], inplace=True)


Ytrain = train["Y"]
Xtrain = train.drop(columns=["Y"])
Yvalid = valid["Y"]
Xvalid = valid.drop(columns=["Y"])
Ytest = test["Y"]
Xtest = test.drop(columns=["Y"])




# 建模

## 绘制效果图像函数
def PotRoc(Y, preds, pos_label=1):
    """绘制roc曲线"""
    fpr, tpr, thresholds = metrics.roc_curve(Y, preds, pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw=2
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % auc)
    plt.plot([0,1],[0,1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc='lower right')
    plt.show()


def recall_precision_f1(Y, preds, k=10):
    """通过阈值变化绘制recall、precision、f1的变化图"""
    recall_column, precision_column, f1_column = [], [], []
    for i in range(k):
        Ypred = preds.copy()
        Ypred[preds > i/k] = 1
        Ypred[Ypred != 1] = 0
        recall_column.append(recall_score(Y, Ypred))
        precision_column.append(precision_score(Y, Ypred))
        f1_column.append(f1_score(Y, Ypred))
    plt.figure()
    plt.plot(range(k), recall_column, color="darkorange", label="recall_score")
    plt.plot(range(k), precision_column, color="navy", label="precision_score")
    plt.plot(range(k), f1_column, color="red", label="f1_score")
    plt.xlabel("thresholds")
    plt.ylabel("score")
    plt.title("recall_precision_f1 by thresholds")
    plt.legend(loc="best")
    plt.show()

## xgboost 建模
dtrain = xgb.DMatrix(Xtrain, Ytrain)  # 这里需要确保Xtrain和Ytrain的数据类型不能为str, 否则会报错，可以在Xtrain后添加".apply(pd.to_numeric)"或者定义dtype解决
dvalid = xgb.DMatrix(Xvalid, Yvalid)
dtest = xgb.DMatrix(Xtest, Ytest)

params = {'objective':"binary:logistic", "subsample":0.7, "eval_metric":"auc", "seed":2022}

xgbM = xgb.train(params=params, dtrain=dtrain, num_boost_round=200, verbose_eval=10)

preds = xgbM.predict(dtest)
# 查看效果
PotRoc(Y=Ytest, preds=preds)
recall_precision_f1(Y=Ytest, preds=preds, k=10)

# 查看特征重要性
fig, ax = plt.subplots(figsize=(6, 20))
xgb.plot_importce(xgbM, max_num_features=70, ax=ax)
