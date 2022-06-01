from sklearn import svm
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def my_svm_Images(Data,pca_components=None):
    X_train = Data['Images_train']
    Labels_train = Data['Labels_train']
    X_val = Data['Images_val']
    Labels_val = Data['Labels_val']
    X_test = Data['Images_test']
    Labels_test = Data['Labels_test']

    X_train = StandardScaler().fit_transform(X_train)
    pca=PCA(n_components=pca_components, copy=True, whiten=False)
    X_train=pca.fit_transform(X_train)

    X_test = StandardScaler().fit_transform(X_test)
    X_test=pca.transform(X_test)

    X_train_pos = X_train[Labels_train == 1, :]
    X_train_neg = X_train[Labels_train == 0, :]
    Labels_train_pos = Labels_train[Labels_train == 1]
    Labels_train_neg = Labels_train[Labels_train == 0]

    numpos = sum(Labels_train == 1)
    numneg = sum(Labels_train == 0)

    numclassifer = int(numneg / numpos)
    SVM_bagging = []

    for id_calssifer in range(numclassifer):
        ### 每个分类器的训练数据都不一样，但是是平衡的
        x_train=np.concatenate((X_train_pos,X_train_neg[id_calssifer*numpos:(id_calssifer+1)*numpos,:]),0)
        y_train=np.concatenate((Labels_train_pos,Labels_train_neg[id_calssifer*numpos:(id_calssifer+1)*numpos]),0)
        x_test=X_test
        y_test = Labels_test
        classifer = svm.SVC(probability=True)
        classifer.fit(x_train, y_train)
        SVM_bagging.append(classifer)

        print("*" * 30 + " 单个SVM分类器准确率 " + "*" * 30)
        print(classifer.score(x_test,y_test))

    joblib.dump(SVM_bagging, './Images/SVM_bagging.pkl')
    joblib.dump(pca, './Images/pca.pkl')
    return SVM_bagging,pca

