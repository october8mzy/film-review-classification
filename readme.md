

该项目完成了电影评语分类功能，分类准确率达到74%。
项目内共有4个文件（夹），其中包括：
	数据集
	train_test.py  
	data_helpers.py
	text_cnn.py

数据集：
	总词条数19417，语句10000条。积极、消极评论各占一半。
	其中的90%作为训练集，剩下的10%作为测试集（保证测试集中正、负样本数量相当）

实现方法：
	应用单隐藏层卷积神经网络完成评语分类的功能。
