# Kaggle競賽：Titanic - Machine Learning from Disaster

## 組員
10946012 李姍珊
10946013 趙晴
10946025 高培芮

## 成績
![](https://hackmd.io/_uploads/S1dzui7w2.png)
![](https://hackmd.io/_uploads/SJTZusXvh.png)

## 摘要
本組選擇鐵達尼號生存預測為對這件事故有初略的了解，且是歷史上重要的事件之一。因此想透過乘客資訊像是性別、年齡…等去預估乘客是否會在鐵達尼號沉沒意外中生存下來。

## 介紹（研究背景及研究目的）
鐵達尼號沉沒事故是當時北大西洋發生的最大著名船難，當時與冰山擦撞前，已收到6次海冰警告，船行駛的速度快速，看到冰山已經為時已晚，無法及時轉向，16個水密隔艙中的5個進水，而鐵達尼號的設計只能承受4個水密隔艙進水因此沉沒，此次災難造成1514人死亡。因此我們想藉由此事件，透過訓練數據分析生還人數，且能預防未來相似的事件發生。

## 資料集介紹(含資料特徵)及資料集來源
此競賽共有2份資料集，分別為train(用來訓練模型)及test(要求預測結果)，還有一份data(合併train與test的資料)，以利接下來的處理。

| 特徵名稱 | 特徵定義 | Key |
|:-------- | -------- | -------- |
| PassengerId     | 乘客編號     |      |
| Survived     | 是否倖存     | 1:是 / 0:否     |
| Pclass     | 船票等級     | 1:最高 / 2:中等 / 3:最低     |
| Name     | 姓名     |      |
| Sex     | 性別     |      |
| Age     | 年齡     |      |
| SibSp     | 同為兄弟姐妹或配偶的數目     |      |
| Parch     | 同為家族父母及小孩的數目     |      |
| Ticket     | 船票編號     |      |
| Fare     | 船票價格     |      |
| Cabin     | 船艙號碼     |      |
| Embarked     | 登船點     | C=Cherbourg、Q=Queenstown、S=Southampton|

### 船上的乘客各年齡層的男女比例
![](https://hackmd.io/_uploads/r1-EKs7D3.png)

## 資料預處理
![](https://hackmd.io/_uploads/By2IDi7P3.png)

由以上合併資料結果來看，得知：
* Age 缺1309 - 1046 = 263筆資料
* Fare 缺1309 - 1308 = 1筆資料
* Cabin 缺1309 - 295 = 1014筆資料
* Embarked 缺1309 - 1307 = 2筆資料

因此先**填補缺漏值**：
* Age：我們以乘客稱呼(Miss.、Ms.等…)來區分，並分別填上平均年齡。
* Fare：因只有缺1筆資料，所以直接用平均值填入。
* Cabin：因缺值太多，目前選擇先不作為特徵使用。
* Embarked：從分析上，發現C港口的乘客大多是P1等級的票，因此選擇填入C值。

**檢視非數值欄位**： <p>
Name欄有2筆是重複的，而Sex欄只有Male/Female這2種值，其中以Male最多，有843位。
![](https://hackmd.io/_uploads/SkDgAWGk6.png)

## 機器學習或深度學習方式（使用何種方式）
根據測試結果，本組使用隨機森林來訓練模型
![](https://hackmd.io/_uploads/Hk5GwsXw2.png)


## 研究結果及討論（含模型評估與改善）
![](https://hackmd.io/_uploads/Bkqrwjmv3.png)

根據以上測試結果，發現XGB、GBDT、LGBM、邏輯斯回歸、隨機森林等都有蠻高的分數，因此我們分別將分數超過0.85的演算法上傳至Kaggle評分，得出以下分數：

![](https://hackmd.io/_uploads/HJ8_wimD2.png)


## 結論
一開始在測試階段時，我們有嘗試自己額外添加Feature，但實際上傳評分時效果並不佳，後來使用原始Feature與測試多個演算法過程中，我們得出LightGBM是最好的，但上傳至Kaggle卻是RandomForest最好，我們目前推測可能是因為資料的過度擬和才會造成此問題，必須再對資料進行更好的處理，從而來解決問題並獲得更好的分數。

## 參考文獻