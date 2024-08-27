# pytorch_bert_intent_classification_and_slot_filling
基于pytorch的中文意图识别和槽位填充

# 说明
基本思路就是：分类+序列标注（命名实体识别）同时训练。<br>
使用的预训练模型：hugging face上的chinese-bert-wwm-ext<br>

1、安装依赖
```python
pytorch==1.6+
transformers==4.5.0
seqeval
```

2、先按照data/train.json和data/test.json的格式准备好数据，然后运行data/preprocess.py，生成训练和测试的文件。<br>

3、修改配置文件config.py，如果需要基于chinese-bert-wwm-ext进行微调，设置如下参数<br>
```python
do_train = True
do_eval = True
do_test = True
do_save = True
do_predict = True
load_model = False
```
如果已经训练好了模型，需要推理，设置如下参数<br>
```python
do_train = False
do_eval = False
do_test = False
do_save = False
do_predict = True
load_model = True
```

4、运行main.py
```python
python main.py
```

5、基于数据集以及模型评估的结果调整lr、epoch等参数


# 结果
```python
意图识别：
accuracy:0.9767441860465116
precision:0.9767441860465116
recall:0.9767441860465116
f1:0.9767441860465116
              precision    recall  f1-score   support

           0       1.00      0.94      0.97        16
           2       1.00      1.00      1.00         1
           3       1.00      1.00      1.00         4
           4       1.00      1.00      1.00        16
           5       0.00      0.00      0.00         1
           6       1.00      1.00      1.00        22
           7       0.84      0.89      0.86        18
           8       0.98      0.95      0.96        57
           9       1.00      1.00      1.00         2
          10       0.00      0.00      0.00         0
          11       0.00      0.00      0.00         1
          12       0.98      0.99      0.99       327
          13       1.00      1.00      1.00         1
          14       1.00      1.00      1.00         3
          15       1.00      1.00      1.00         1
          17       1.00      1.00      1.00         4
          18       1.00      0.80      0.89         5
          19       1.00      1.00      1.00        14
          21       0.00      0.00      0.00         1
          22       1.00      1.00      1.00        13
          23       1.00      1.00      1.00         9

    accuracy                           0.98       516
   macro avg       0.80      0.79      0.79       516
weighted avg       0.97      0.98      0.97       516

槽位填充：
accuracy:0.9366942909760589
precision:0.8052708638360175
recall:0.8461538461538461
f1:0.8252063015753938
                   precision    recall  f1-score   support

             Dest       1.00      1.00      1.00         7
              Src       1.00      0.86      0.92         7
             area       1.00      0.25      0.40         4
           artist       0.89      1.00      0.94         8
       artistRole       1.00      1.00      1.00         2
           author       1.00      1.00      1.00        13
         category       0.73      0.90      0.81        42
             code       0.71      0.83      0.77         6
          content       0.89      0.94      0.91        17
    datetime_date       0.72      0.95      0.82        19
    datetime_time       0.58      0.64      0.61        11
         dishName       0.84      0.88      0.86        74
        dishNamet       0.00      0.00      0.00         1
          dynasty       1.00      1.00      1.00        11
      endLoc_area       0.00      0.00      0.00         2
      endLoc_city       0.96      1.00      0.98        43
       endLoc_poi       0.62      0.73      0.67        11
  endLoc_province       0.00      0.00      0.00         1
          episode       1.00      1.00      1.00         1
             film       0.00      0.00      0.00         1
       ingredient       0.53      0.62      0.57        16
          keyword       0.88      0.88      0.88        25
    location_area       0.00      0.00      0.00         2
    location_city       0.40      1.00      0.57         4
     location_poi       0.36      0.57      0.44         7
location_province       0.00      0.00      0.00         3
             name       0.80      0.88      0.84       182
       popularity       0.00      0.00      0.00         5
       queryField       1.00      1.00      1.00         2
     questionWord       0.00      0.00      0.00         1
         receiver       1.00      1.00      1.00         4
         relIssue       0.00      0.00      0.00         1
       scoreDescr       0.00      0.00      0.00         1
             song       0.86      0.80      0.83        15
   startDate_date       0.93      0.93      0.93        15
   startDate_time       0.00      0.00      0.00         1
    startLoc_area       0.00      0.00      0.00         1
    startLoc_city       0.95      0.97      0.96        38
     startLoc_poi       0.00      0.00      0.00         1
         subfocus       0.00      0.00      0.00         1
              tag       0.40      0.40      0.40         5
           target       1.00      1.00      1.00        12
     teleOperator       0.00      0.00      0.00         1
          theatre       0.50      0.50      0.50         2
        timeDescr       0.00      0.00      0.00         2
        tvchannel       0.74      0.81      0.77        21
        yesterday       0.00      0.00      0.00         1

        micro avg       0.81      0.85      0.83       650
        macro avg       0.52      0.54      0.52       650
     weighted avg       0.79      0.85      0.81       650

=================================
```

# 补充
上述实验只是基于自己的思路所做的，有些同学需要了解相关知识，这里附上：<br>
[意图识别和槽位填充综述](http://cn.arxiv.org/abs/2101.08091) <br>
[利用bert进行意图识别和槽位填充](https://zhuanlan.zhihu.com/p/415530908)
