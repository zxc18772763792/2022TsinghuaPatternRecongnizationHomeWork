# -HIV-
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
Id_test为想要预测的样本在总样本集中的索引，如123
### 3.训练并得到相关的分析结果
在conda命令行输入：
```
python 绝对路径\Codes\main.py total_ratio train_ratio val_ratio test_ratio
```
在提交的结果中使用total_ratio=0.1 train_ratio0.7 val_ratio=0.2 test_ratio=0.1
