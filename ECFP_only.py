import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def enrtopy_np(class1):
    p1=sum(class1==1)/len(class1)
    if p1==0 or p1==1:
        ent=0
    else:
        ent=p1*np.log2(p1)+(1-p1)*np.log2(1-p1)
        ent=-ent
    return ent

def H_split(class1,class2):
    if len(class1)>=1 and len(class2)>=1:
        ent1=enrtopy_np(class1)
        ent2 = enrtopy_np(class2)
        H=(ent1*len(class1)+ent2*len(class2))/(len(class1)+len(class2))
    else:
        H=0

    return H

def feature_slection(Data,numfeature_selected=256):
    ### Data is pos-neg balanced
    ECFP_train=Data['ECFP_train']
    Labels_train=Data['Labels_train']
    ### select features depends on the Information gain of each feature
    numfeatures=ECFP_train.shape[1]
    numsamples=ECFP_train.shape[0]
    Ent=np.zeros(numfeatures)
    for id_feature in range(numfeatures):
        feature=ECFP_train[:,id_feature]
        class1=Labels_train[feature==0]
        class2=Labels_train[feature==1]
        Ent[id_feature]=H_split(class1,class2)

    order=Ent.argsort()
    selected_feature_id=order[-numfeature_selected-1:-1]
    return selected_feature_id

def my_forests_ECFP(Data,numfeature_selected=256):
    X_train = Data['ECFP_train']
    Labels_train = Data['Labels_train']
    X_val = Data['ECFP_val']
    Labels_val = Data['Labels_val']
    X_test = Data['ECFP_test']
    Labels_test = Data['Labels_test']

    X_train_pos = X_train[Labels_train == 1, :]
    X_train_neg = X_train[Labels_train == 0, :]
    Labels_train_pos = Labels_train[Labels_train == 1]
    Labels_train_neg = Labels_train[Labels_train == 0]

    numpos = sum(Labels_train == 1)
    numneg = sum(Labels_train == 0)

    numclassifer = int(numneg / numpos)
    RF_bagging = []
    selected_feature_id=np.zeros([numfeature_selected,numclassifer]).astype(int)
    for id_calssifer in range(numclassifer):
        ### 每个分类器的训练数据都不一样，但是是平衡的
        x_train=np.concatenate((X_train_pos,X_train_neg[id_calssifer*numpos:(id_calssifer+1)*numpos,:]),0)
        y_train=np.concatenate((Labels_train_pos,Labels_train_neg[id_calssifer*numpos:(id_calssifer+1)*numpos]),0)
        Data_mini={
            'ECFP_train':x_train,
            'Labels_train':y_train,
            'ECFP_val':X_val,
            'Labels_val':Labels_val,
            'ECFP_test':X_test,
            'Labels_test':Labels_test
        }
        selected_feature_id[:,id_calssifer]=feature_slection(Data_mini, numfeature_selected=numfeature_selected).astype(int)
        x_train =x_train[:,selected_feature_id[:,id_calssifer]]
        x_test=X_test[:,selected_feature_id[:,id_calssifer]]
        y_test = Labels_test

        rf = RandomForestClassifier(n_jobs=-1)
        # 网格搜索
        # n_estimators: 决策树数目
        # max_depth: 树最大深度
        param = {
            "n_estimators": [120, 200, 300, 500, 800, 1200],
            "max_depth": [5, 8, 15, 25, 30, 50]
        }
        # 2折交叉验证
        grid_search = GridSearchCV(rf, param_grid=param, cv=2)
        grid_search.fit(x_train, y_train)
        final_rf = grid_search.best_estimator_
        RF_bagging.append(final_rf)

        print("*" * 30 + " 单个随机森林分类器准确率 " + "*" * 30)
        print(grid_search.score(x_test, y_test))
    joblib.dump(RF_bagging, './ECFP/RF_bagging.pkl')
    joblib.dump(selected_feature_id, './ECFP/selected_feature_id.pkl')

    return RF_bagging,selected_feature_id


