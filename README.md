# -有机小分子HIV抑制效果预测-
2022年春季学期清华大学模式识别（张长水）课程大作业-有机小分子HIV抑制效果预测
## Code
### 1.环境配置
创建虚拟环境PRproject2，并激活
```
conda create --name PRproject2 python=3.9
conda activate PRproject2 
```
将requirements.txt文件放到虚拟环境下，一般为（C:\Users\Username\）
pip download -r requirements.txt

### 2.Ensemble前最好的分类器单个样本的测试
在conda命令行输入：
```
python 绝对路径\Codes\SingleTest.py Id_test
```
Id_test为想要预测的样本在总样本集中的索引，如123，测试结果如下：
![singleTest](https://user-images.githubusercontent.com/54254118/171396429-95dc7bf9-a66b-47bf-afe4-9c336ad8bb28.png)

### 3.训练并得到相关的分析结果
在conda命令行输入：
```
python 绝对路径\Codes\main.py total_ratio train_ratio val_ratio test_ratio
```
例如在提交的结果中使用total_ratio=0.1 train_ratio0.7 val_ratio=0.2 test_ratio=0.1，可用命令：
```
python 绝对路径\Codes\main.py 0.1 0.7 0.2 0.1
```
训练和分析结束后，会得到名为“Indicator_XXX.xlsx”的Excel表格以及训练好的模型，测试集的预测结果Labels_pred、预测分数Socres_pred与Precision、Recall、PR—AUC、F1分数均在Excel中存储。
![Indicator1](https://user-images.githubusercontent.com/54254118/171396985-d6d82e48-5cc6-4829-a6dd-daa04a5b29a4.png)
每个Excel包含4张表：“Labels_pred”、“Scores_pred”、“Indicators”、“Ensemble Confusion Matrix”：
![Indicator2](https://user-images.githubusercontent.com/54254118/171397790-b1df608b-4c44-48ed-913f-0d9d6bd42469.png)

### 4.各部分代码说明
#### 1.ProcessData.py
数据预处理，从"SMILES.npy"、"ECFP.npy"、"Image.npy"、"Label.npy"中读取数据，并按照正负样本的比例从中选择total_ratio的样本构建mini数据集，mini数据集按照train_ratio:val_ratio:test_ratio划分为训练集、验证集和测试集。后续所有的训练和测试均在mini数据集上完成。为了保证训练模型的一致性，上传的代码中数据的抽取按照索引从小到大按顺序抽取，如需实用请调整为随机采样。该函数返回一个字典Data。
#### 2.SMILES_only.py
transformer_SMILES_origin函数训练单个DislitBert分类器，transfomer_bagging函数用Bagging的方式集成训练好的DislitBert弱分类器。DislitBert预训练头基于HuggingFace的相关工作。
#### 3.ECFP_only.py
feature_slection函数根据信息增益选择指定个特征对ECFP特征进行降维。my_forests_ECFP函数利用sklearn包训练随机森林（RF）弱分类器并进行Bagging集成。
#### 4.Images_only.py
my_svm_Images函数对Image特征进行Z-score标准化后利用PCA计数降维到指定的维度，再利用sklearn包训SVM弱分类器并进行Bagging集成。
#### 5.CNN_Images.py
定义了网络结构CNN类，构建数据集myDataset类，myCNN_Images函数训练单个的CNN弱分类器，CNNs_bagging训练CNN分类器并进行Bagging集成。
#### 6.Analysis.py
analysis_transformers、analysis_RFs、analysis_SVMs、analysis_CNNs、analysis_Final分别计算了几个模型单独在测试集上的准确率（Accuracy）、精确率（Precision）、召回率（Recall）、精确率-召回率曲线下的面积（PR-AUC）和F1分数（F1 Score），返回结果为一个字典Indicator，包含了几个模型在测试集上的表现。
#### 7.SingleTest.py
命令行输入一个想要测试的样本在样本集中的索引，如123，返回集成前最好的单个DislitBert分类器、RF分类器、SVM分类器、CNN分类器的分类结果和相应的预测分数。
#### 8.main.py
训练以上模型并将模型在测试集上的表现写入"Indicator_XXX.xlsx"的Excel表格中。
