from CNN_Images import myDataset
from SMILES_only import  construct_test_dataset
import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional
import torch.utils.data as data
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
import xlsxwriter as xw
import os

def Indicator_toExcel(Indicator, fileName,Real_Label):  # xlsxwriter库储存数据到excel
    workbook = xw.Workbook(fileName)  # 创建工作簿
    ### 写Labels_pred_all
    worksheet1 = workbook.add_worksheet("Labels_pred_all")  # 创建子表
    worksheet1.activate()  # 激活表
    numclassifers=Indicator['Labels_pred_all'].shape[0]
    numsamples=Indicator['Labels_pred_all'].shape[1]
    title=['Classifer'+str(s+1) for s in range(numclassifers)]
    title.append('Ensemble')
    title.append('Real')
    worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头
    i = 2  # 从第二行开始写入数据
    for j in range(numsamples):
        insertData = Indicator['Labels_pred_all'][:,j]
        row = 'A' + str(i)
        worksheet1.write_row(row, insertData)
        i += 1
    worksheet1.write_column(1,numclassifers,Indicator['Labels_pred_ensemble'])
    worksheet1.write_column(1,numclassifers+1,Real_Label)

    ### 写 Scores_pred_all
    worksheet2 = workbook.add_worksheet("Scores_pred_all")  # 创建子表
    worksheet2.activate()  # 激活表
    title=['Classifer'+str(s+1) for s in range(numclassifers)]
    title.append('Ensemble')
    worksheet2.write_row('A1', title)  # 从A1单元格开始写入表头
    i = 2  # 从第二行开始写入数据
    for j in range(numsamples):
        insertData = Indicator['Scores_pred_all'][:,j,1]
        row = 'A' + str(i)
        worksheet2.write_row(row, insertData)
        i += 1
    worksheet2.write_column(1, numclassifers, Indicator['Ensemble_probs'][:,1])

    ### 写 其他指标
    worksheet3 = workbook.add_worksheet("Indicators")  # 创建子表
    worksheet3.activate()  # 激活表
    title = ['Classifer','Accuracy','Precision','Recall','PR_AUC','F1']  # 设置表头
    worksheet3.write_row('A1', title)  # 从A1单元格开始写入表头
    i = 2  # 从第二行开始写入数据
    for j in range(numclassifers):
        insertData = [str(j + 1), Indicator['ACC_all'][j], Indicator['Precisions_all'][j], Indicator['Recall_all'][j],
                      Indicator['PR_AUC_all'][j], Indicator['F1_all'][j]]
        row = 'A' + str(i)
        worksheet3.write_row(row, insertData)
        i += 1

    insertData = ['Ensemble', Indicator['Ensemble_acc'], Indicator['Ensemble_precision'], Indicator['Ensemble_recall'],
                  Indicator['Ensemble_auc'], Indicator['Ensemble_f1']]
    row = 'A' + str(i)
    worksheet3.write_row(row, insertData)

    ### 写集成模型的混淆矩阵
    worksheet4 = workbook.add_worksheet("Ensemble_CM")  # 创建子表
    worksheet4.activate()  # 激活表
    worksheet4.write_row('A1', ['Confusion Matrix of Ensemble Model'])
    title=['Pred Positive','Pred Negative']
    worksheet4.write_row('B2', title)  # 从A1单元格开始写入表头
    worksheet4.write_column(2,0,['Real Positive','Real Negative'])
    # TP = confusion_matrix[0, 0];
    # FN = confusion_matrix[0, 1]
    # FP = confusion_matrix[1, 0];
    # TN = confusion_matrix[1, 1]
    worksheet4.write(2,1,Indicator['Ensemble_CM'][0,0]) #TP
    worksheet4.write(3,1,Indicator['Ensemble_CM'][1,0]) # FP
    worksheet4.write(2,2,Indicator['Ensemble_CM'][0,1]) #FN
    worksheet4.write(3,2,Indicator['Ensemble_CM'][1,1]) # TN
    workbook.close()  # 关闭表

def Final_Indicator_toExcel(Indicator, fileName,Real_Label):  # xlsxwriter库储存数据到excel
    workbook = xw.Workbook(fileName)  # 创建工作簿
    numsamples=len(Real_Label)
    ### 写Labels_pred_final
    worksheet1 = workbook.add_worksheet("Labels_pred_ensemble")  # 创建子表
    worksheet1.activate()  # 激活表
    title=['Ensemble']
    title.append('Real')
    worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头
    worksheet1.write_column(1,0,Indicator['Labels_pred_ensemble'])
    worksheet1.write_column(1,1,Real_Label)

    ### 写 Final_probs
    worksheet2 = workbook.add_worksheet("Scores_final")  # 创建子表
    worksheet2.activate()  # 激活表
    title=['0','1']
    worksheet2.write_row('A1', title)  # 从A1单元格开始写入表头
    i = 2  # 从第二行开始写入数据
    for j in range(numsamples):
        insertData = Indicator['Final_probs'][j,:]
        row = 'A' + str(i)
        worksheet2.write_row(row, insertData)
        i += 1

    ### 写 其他指标
    worksheet3 = workbook.add_worksheet("Indicators")  # 创建子表
    worksheet3.activate()  # 激活表
    title = ['Accuracy','Precision','Recall','PR_AUC','F1']  # 设置表头
    worksheet3.write_row('A1', title)  # 从A1单元格开始写入表头
    i = 2  # 从第二行开始写入数据
    insertData = [Indicator['Final_acc'], Indicator['Final_precision'], Indicator['Final_recall'],
                  Indicator['Final_auc'], Indicator['Final_f1']]
    row = 'A' + str(i)
    worksheet3.write_row(row, insertData)

    ### 写集成模型的混淆矩阵
    worksheet4 = workbook.add_worksheet("Final_CM")  # 创建子表
    worksheet4.activate()  # 激活表
    worksheet4.write_row('A1', ['Confusion Matrix of Ensemble Model'])
    title=['Pred Positive','Pred Negative']
    worksheet4.write_row('B2', title)  # 从A1单元格开始写入表头
    worksheet4.write_column(2,0,['Real Positive','Real Negative'])
    # TP = confusion_matrix[0, 0];
    # FN = confusion_matrix[0, 1]
    # FP = confusion_matrix[1, 0];
    # TN = confusion_matrix[1, 1]
    worksheet4.write(2,1,Indicator['Final_CM'][0,0]) #TP
    worksheet4.write(3,1,Indicator['Final_CM'][1,0]) # FP
    worksheet4.write(2,2,Indicator['Final_CM'][0,1]) #FN
    worksheet4.write(3,2,Indicator['Final_CM'][1,1]) # TN
    workbook.close()  # 关闭表

def analysis_transformers(Data):
    ###### SMILES Only ##########
    # Single Transformer Classifer
    Models_bagging=joblib.load('./SMILES/Transformers_bagging.pkl')
    tokenizer=joblib.load('./SMILES/tokenizer.pkl')

    numclassifer=len(Models_bagging)
    numsamples_test=len(Data['SMILES_org_test'].tolist())
    pt_batch=construct_test_dataset(Data, tokenizer)

    Labels_pred = np.zeros([numclassifer,numsamples_test ])
    Scores_pred=np.zeros([numclassifer,numsamples_test,2])
    Labels_pred_ensemble=np.zeros(numsamples_test)
    Precisions_transformer=np.zeros(numclassifer)
    Recall_transformer=np.zeros(numclassifer)
    PR_AUC_transformer=np.zeros(numclassifer)
    F1_transformer=np.zeros(numclassifer)
    ACC_transformer=np.zeros(numclassifer)
    Confusion_Matrix_transformer=np.zeros([numclassifer,2,2])
    print("*"*5+"  Accuracy  "+"  Precision  "+"Recall  "+"PR_AUC  "+"F1  "+"*"*5)
    for id_classifer, classifer in enumerate(Models_bagging):
        with torch.no_grad():
            classifer.eval()
            output = classifer(**pt_batch)
            pt_predictions = nn.functional.softmax(output.logits, dim=-1)
            Scores_pred[id_classifer,:,:]=pt_predictions.detach().numpy()
            single_acc=0
            for ii in range(pt_predictions.shape[0]):
                pred = pt_predictions[ii, :]
                label_real = Data['Labels_test'].tolist()[ii]
                if pred[0] >= pred[1]:
                    Labels_pred[id_classifer,ii] = 0
                else:
                    Labels_pred[id_classifer,ii] = 1
                if label_real==Labels_pred[id_classifer,ii]:
                    single_acc= single_acc + 1
        single_acc= single_acc / pt_predictions.shape[0]
        single_precision, single_recall, _ = precision_recall_curve(Data['Labels_test'].tolist(), pt_predictions[:,1].detach().numpy().tolist())
        single_f1, single_auc = f1_score(Data['Labels_test'].tolist(), Labels_pred[id_classifer,:].tolist()), auc(single_recall, single_precision)
        single_CM = confusion_matrix(Data['Labels_test'].tolist(), Labels_pred[ id_classifer,:].tolist())
        single_precision = precision_score(Data['Labels_test'].tolist(), Labels_pred[id_classifer,:].tolist(), pos_label=1)
        single_recall = recall_score(y_true=Data['Labels_test'].tolist(), y_pred=Labels_pred[id_classifer,:].tolist(), pos_label=1)

        Precisions_transformer[id_classifer]=single_precision
        Recall_transformer[id_classifer]=single_recall
        PR_AUC_transformer[id_classifer]=single_auc
        F1_transformer[id_classifer]=single_f1
        ACC_transformer[id_classifer]=single_acc
        Confusion_Matrix_transformer[id_classifer,:,:]=single_CM

        print(" "+str(id_classifer)+"    %.4f    %.4f    %.4f    %.4f    %.4f"%(single_acc,single_precision,single_recall,single_auc,single_f1))

    Labels_pred_ensemble=Labels_pred.sum(axis=0)
    acc = 0
    for ii in range(numsamples_test):
        label_real = Data['Labels_test'].tolist()[ii]
        label_pred = Labels_pred_ensemble[ii]
        if label_pred >= numclassifer / 2:
            Labels_pred_ensemble[ii] = 1
        else:
            Labels_pred_ensemble[ii] = 0

        if label_real == Labels_pred_ensemble[ii]:
            acc = acc + 1
    acc = acc / numsamples_test

    Ensemble_probs=Scores_pred.sum(axis=0)/numclassifer
    Ensemble_precision, Ensemble_recall, _ = precision_recall_curve(Data['Labels_test'].tolist(), Ensemble_probs[:, 1].tolist())
    Ensemble_f1, Ensemble_auc = f1_score(Data['Labels_test'].tolist(), Labels_pred_ensemble.tolist()), auc(Ensemble_recall, Ensemble_precision)
    fig = pyplot.figure()
    pyplot.plot(Ensemble_recall, Ensemble_precision, marker='.', label='Transformer')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    fig.savefig('Transformer_bagging_pr_curve.png')
    pyplot.close()

    Ensemble_CM = confusion_matrix(Data['Labels_test'].tolist(), Labels_pred_ensemble.tolist())
    Ensemble_precision = precision_score(Data['Labels_test'].tolist(), Labels_pred_ensemble.tolist(), pos_label=1)
    Ensemble_recall = recall_score(y_true=Data['Labels_test'].tolist(), y_pred=Labels_pred_ensemble.tolist(), pos_label=1)

    print(" Ensembled %.4f    %.4f    %.4f    %.4f    %.4f"%(acc,Ensemble_precision,Ensemble_recall,Ensemble_auc,Ensemble_f1))

    Indicator={
        "Labels_pred_all":Labels_pred,
        "Scores_pred_all":Scores_pred,
        "Precisions_all":Precisions_transformer,
        "Recall_all":Recall_transformer,
        "PR_AUC_all":PR_AUC_transformer,
        "F1_all":F1_transformer,
        "ACC_all":ACC_transformer,
        "Confusion_Matrix_all":Confusion_Matrix_transformer,
        "Labels_pred_ensemble":Labels_pred_ensemble,
        "Ensemble_CM":Ensemble_CM,
        "Ensemble_precision":Ensemble_precision,
        "Ensemble_recall":Ensemble_recall,
        "Ensemble_f1":Ensemble_f1,
        "Ensemble_auc":Ensemble_auc,
        "Ensemble_acc":acc,
        "Ensemble_probs":Ensemble_probs
    }
    joblib.dump(Indicator,'./SMILES/Indicator.pkl')
    return Indicator

def analysis_RFs(Data):
    ###### ECFP Only ##########
    # Single RF Classifer
    RFs_bagging = joblib.load('./ECFP/RF_bagging.pkl')
    selected_feature_id = joblib.load('./ECFP/selected_feature_id.pkl')

    X_test = Data['ECFP_test']
    Labels_test = Data['Labels_test']
    numclassifer = len(RFs_bagging)
    numsamples_test = len(X_test)

    Labels_pred = np.zeros([numclassifer, numsamples_test])
    Scores_pred = np.zeros([numclassifer, numsamples_test, 2])
    Labels_pred_ensemble = np.zeros(numsamples_test)
    Precisions_all = np.zeros(numclassifer)
    Recall_all = np.zeros(numclassifer)
    PR_AUC_all = np.zeros(numclassifer)
    F1_all = np.zeros(numclassifer)
    ACC_all = np.zeros(numclassifer)
    Confusion_Matrix_all = np.zeros([numclassifer, 2, 2])
    print("*" * 5 + "  Accuracy  " + "  Precision  " + "Recall  " + "PR_AUC  " + "F1  " + "*" * 5)
    for id_classifer, classifer in enumerate(RFs_bagging):
        Scores_pred[id_classifer, :, :] = classifer.predict_proba(X_test[:, selected_feature_id[:, id_classifer]])
        Labels_pred[id_classifer, :] = classifer.predict(X_test[:, selected_feature_id[:, id_classifer]])

        single_precision, single_recall, _ = precision_recall_curve(Labels_test.tolist(),
                                                                    Scores_pred[id_classifer, :, 1].tolist())
        single_f1, single_auc = f1_score(Labels_test.tolist(), Labels_pred[id_classifer, :].tolist()), auc(
            single_recall, single_precision)
        single_CM = confusion_matrix(Labels_test.tolist(), Labels_pred[id_classifer, :].tolist())
        single_precision = precision_score(Labels_test.tolist(), Labels_pred[id_classifer, :].tolist(),
                                           pos_label=1)
        single_recall = recall_score(y_true=Labels_test.tolist(), y_pred=Labels_pred[id_classifer, :].tolist(),
                                     pos_label=1)
        single_acc = accuracy_score(y_true=Labels_test.tolist(), y_pred=Labels_pred[id_classifer, :].tolist())

        Precisions_all[id_classifer] = single_precision
        Recall_all[id_classifer] = single_recall
        PR_AUC_all[id_classifer] = single_auc
        F1_all[id_classifer] = single_f1
        ACC_all[id_classifer] = single_acc
        Confusion_Matrix_all[id_classifer, :, :] = single_CM

        print(" " + str(id_classifer) + "    %.4f    %.4f    %.4f    %.4f    %.4f" % (
            single_acc, single_precision, single_recall, single_auc, single_f1))

    Labels_pred_ensemble = Labels_pred.sum(axis=0)
    acc = 0
    for ii in range(numsamples_test):
        label_real = Labels_test.tolist()[ii]
        label_pred = Labels_pred_ensemble[ii]
        if label_pred >= numclassifer / 2:
            Labels_pred_ensemble[ii] = 1
        else:
            Labels_pred_ensemble[ii] = 0

        if label_real == Labels_pred_ensemble[ii]:
            acc = acc + 1
    acc = acc / numsamples_test

    Ensemble_probs = Scores_pred.sum(axis=0) / numclassifer
    Ensemble_precision, Ensemble_recall, _ = precision_recall_curve(Labels_test.tolist(),
                                                                    Ensemble_probs[:, 1].tolist())
    Ensemble_f1, Ensemble_auc = f1_score(Labels_test.tolist(), Labels_pred_ensemble.tolist()), auc(
        Ensemble_recall, Ensemble_precision)
    fig = pyplot.figure()
    pyplot.plot(Ensemble_recall, Ensemble_precision, marker='.', label='RF')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    fig.savefig('RF_bagging_pr_curve.png')
    pyplot.close()

    Ensemble_CM = confusion_matrix(Labels_test.tolist(), Labels_pred_ensemble.tolist())
    Ensemble_precision = precision_score(Labels_test.tolist(), Labels_pred_ensemble.tolist(), pos_label=1)
    Ensemble_recall = recall_score(y_true=Labels_test.tolist(), y_pred=Labels_pred_ensemble.tolist(),
                                   pos_label=1)

    print(" Ensembled %.4f    %.4f    %.4f    %.4f    %.4f" % (
        acc, Ensemble_precision, Ensemble_recall, Ensemble_auc, Ensemble_f1))

    Indicator = {
        "Labels_pred_all": Labels_pred,
        "Scores_pred_all": Scores_pred,
        "Precisions_all": Precisions_all,
        "Recall_all": Recall_all,
        "PR_AUC_all": PR_AUC_all,
        "F1_all": F1_all,
        "ACC_all": ACC_all,
        "Confusion_Matrix_all": Confusion_Matrix_all,
        "Labels_pred_ensemble": Labels_pred_ensemble,
        "Ensemble_CM": Ensemble_CM,
        "Ensemble_precision": Ensemble_precision,
        "Ensemble_recall": Ensemble_recall,
        "Ensemble_f1": Ensemble_f1,
        "Ensemble_auc": Ensemble_auc,
        "Ensemble_acc": acc,
        "Ensemble_probs": Ensemble_probs
    }
    joblib.dump(Indicator, './ECFP/Indicator.pkl')
    return Indicator

def analysis_SVMs(Data):
    ###### Images Only ##########
    # Single SVM Classifer
    SVMs_bagging = joblib.load('./Images/SVM_bagging.pkl')
    pca = joblib.load('./Images/pca.pkl')
    X_test = Data['Images_test']
    Labels_test = Data['Labels_test']
    X_test = StandardScaler().fit_transform(X_test)
    X_test = pca.transform(X_test)

    numclassifer = len(SVMs_bagging)
    numsamples_test = len(X_test)

    Labels_pred = np.zeros([numclassifer, numsamples_test])
    Scores_pred = np.zeros([numclassifer, numsamples_test, 2])
    Labels_pred_ensemble = np.zeros(numsamples_test)
    Precisions_all = np.zeros(numclassifer)
    Recall_all = np.zeros(numclassifer)
    PR_AUC_all = np.zeros(numclassifer)
    F1_all = np.zeros(numclassifer)
    ACC_all = np.zeros(numclassifer)
    Confusion_Matrix_all = np.zeros([numclassifer, 2, 2])
    print("*" * 5 + "  Accuracy  " + "  Precision  " + "Recall  " + "PR_AUC  " + "F1  " + "*" * 5)
    for id_classifer, classifer in enumerate(SVMs_bagging):
        Scores_pred[id_classifer, :, :] = classifer.predict_proba(X_test)
        Labels_pred[id_classifer, :] = classifer.predict(X_test)

        single_precision, single_recall, _ = precision_recall_curve(Labels_test.tolist(),
                                                                    Scores_pred[id_classifer, :, 1].tolist())
        single_f1, single_auc = f1_score(Labels_test.tolist(), Labels_pred[id_classifer, :].tolist()), auc(
            single_recall, single_precision)
        single_CM = confusion_matrix(Labels_test.tolist(), Labels_pred[id_classifer, :].tolist())
        single_precision = precision_score(Labels_test.tolist(), Labels_pred[id_classifer, :].tolist(),
                                           pos_label=1)
        single_recall = recall_score(y_true=Labels_test.tolist(), y_pred=Labels_pred[id_classifer, :].tolist(),
                                     pos_label=1)
        single_acc = accuracy_score(y_true=Labels_test.tolist(), y_pred=Labels_pred[id_classifer, :].tolist())

        Precisions_all[id_classifer] = single_precision
        Recall_all[id_classifer] = single_recall
        PR_AUC_all[id_classifer] = single_auc
        F1_all[id_classifer] = single_f1
        ACC_all[id_classifer] = single_acc
        Confusion_Matrix_all[id_classifer, :, :] = single_CM

        print(" " + str(id_classifer) + "    %.4f    %.4f    %.4f    %.4f    %.4f" % (
            single_acc, single_precision, single_recall, single_auc, single_f1))

    Labels_pred_ensemble = Labels_pred.sum(axis=0)
    acc = 0
    for ii in range(numsamples_test):
        label_real = Labels_test.tolist()[ii]
        label_pred = Labels_pred_ensemble[ii]
        if label_pred >= numclassifer / 2:
            Labels_pred_ensemble[ii] = 1
        else:
            Labels_pred_ensemble[ii] = 0

        if label_real == Labels_pred_ensemble[ii]:
            acc = acc + 1
    acc = acc / numsamples_test

    Ensemble_probs = Scores_pred.sum(axis=0) / numclassifer
    Ensemble_precision, Ensemble_recall, _ = precision_recall_curve(Labels_test.tolist(),
                                                                    Ensemble_probs[:, 1].tolist())
    Ensemble_f1, Ensemble_auc = f1_score(Labels_test.tolist(), Labels_pred_ensemble.tolist()), auc(
        Ensemble_recall, Ensemble_precision)
    fig = pyplot.figure()
    pyplot.plot(Ensemble_recall, Ensemble_precision, marker='.', label='SVM')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    fig.savefig('SVM_bagging_pr_curve.png')
    pyplot.close()

    Ensemble_CM = confusion_matrix(Labels_test.tolist(), Labels_pred_ensemble.tolist())
    Ensemble_precision = precision_score(Labels_test.tolist(), Labels_pred_ensemble.tolist(), pos_label=1)
    Ensemble_recall = recall_score(y_true=Labels_test.tolist(), y_pred=Labels_pred_ensemble.tolist(),
                                   pos_label=1)

    print(" Ensembled %.4f    %.4f    %.4f    %.4f    %.4f" % (
        acc, Ensemble_precision, Ensemble_recall, Ensemble_auc, Ensemble_f1))

    Indicator = {
        "Labels_pred_all": Labels_pred,
        "Scores_pred_all": Scores_pred,
        "Precisions_all": Precisions_all,
        "Recall_all": Recall_all,
        "PR_AUC_all": PR_AUC_all,
        "F1_all": F1_all,
        "ACC_all": ACC_all,
        "Confusion_Matrix_all": Confusion_Matrix_all,
        "Labels_pred_ensemble": Labels_pred_ensemble,
        "Ensemble_CM": Ensemble_CM,
        "Ensemble_precision": Ensemble_precision,
        "Ensemble_recall": Ensemble_recall,
        "Ensemble_f1": Ensemble_f1,
        "Ensemble_auc": Ensemble_auc,
        "Ensemble_acc": acc,
        "Ensemble_probs": Ensemble_probs
    }
    joblib.dump(Indicator, './Images/Indicator_SVM.pkl')
    return Indicator

def analysis_CNNs(Data):
    ###### Images Only ##########
    # Single CNN Classifer

    dataset_test = myDataset(Data['Images_test'], Data['Labels_test'])
    Labels_test = Data['Labels_test']
    test_loader = data.DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, num_workers=0)
    CNNs = joblib.load('./Images/CNNs_bagging.pkl')
    numclassifer = len(CNNs)
    numsamples_test = len(test_loader)

    Labels_pred = np.zeros([numclassifer, numsamples_test])
    Scores_pred = np.zeros([numclassifer, numsamples_test, 2])
    Labels_pred_ensemble = np.zeros(numsamples_test)
    Precisions_all = np.zeros(numclassifer)
    Recall_all = np.zeros(numclassifer)
    PR_AUC_all = np.zeros(numclassifer)
    F1_all = np.zeros(numclassifer)
    ACC_all = np.zeros(numclassifer)
    Confusion_Matrix_all = np.zeros([numclassifer, 2, 2])
    print("*" * 5 + "  Accuracy  " + "  Precision  " + "Recall  " + "PR_AUC  " + "F1  " + "*" * 5)

    for id_classifer, classifer in enumerate(CNNs):
        classifer.eval()
        with torch.no_grad():
            for id_test, (test_img, test_label) in enumerate(test_loader):
                test_output = classifer(test_img)
                Scores_pred[id_classifer, id_test, :] = test_output.detach().numpy()
                Labels_pred[id_classifer, id_test]= torch.max(test_output, 1)[1].data.squeeze().detach().numpy()


        single_precision, single_recall, _ = precision_recall_curve(Labels_test.tolist(),
                                                                    Scores_pred[id_classifer, :, 1].tolist())
        single_f1, single_auc = f1_score(Labels_test.tolist(), Labels_pred[id_classifer, :].tolist()), auc(
            single_recall, single_precision)
        single_CM = confusion_matrix(Labels_test.tolist(), Labels_pred[id_classifer, :].tolist())
        single_precision = precision_score(Labels_test.tolist(), Labels_pred[id_classifer, :].tolist(),
                                           pos_label=1)
        single_recall = recall_score(y_true=Labels_test.tolist(), y_pred=Labels_pred[id_classifer, :].tolist(),
                                     pos_label=1)
        single_acc = accuracy_score(y_true=Labels_test.tolist(), y_pred=Labels_pred[id_classifer, :].tolist())

        Precisions_all[id_classifer] = single_precision
        Recall_all[id_classifer] = single_recall
        PR_AUC_all[id_classifer] = single_auc
        F1_all[id_classifer] = single_f1
        ACC_all[id_classifer] = single_acc
        Confusion_Matrix_all[id_classifer, :, :] = single_CM

        print(" " + str(id_classifer) + "    %.4f    %.4f    %.4f    %.4f    %.4f" % (
        single_acc, single_precision, single_recall, single_auc, single_f1))

    Labels_pred_ensemble = Labels_pred.sum(axis=0)
    acc = 0
    for ii in range(numsamples_test):
        label_real = Labels_test.tolist()[ii]
        label_pred = Labels_pred_ensemble[ii]
        if label_pred >= numclassifer / 2:
            Labels_pred_ensemble[ii] = 1
        else:
            Labels_pred_ensemble[ii] = 0

        if label_real == Labels_pred_ensemble[ii]:
            acc = acc + 1
    acc = acc / numsamples_test

    Ensemble_probs = Scores_pred.sum(axis=0) / numclassifer
    Ensemble_precision, Ensemble_recall, _ = precision_recall_curve(Labels_test.tolist(),
                                                                    Ensemble_probs[:, 1].tolist())
    Ensemble_f1, Ensemble_auc = f1_score(Labels_test.tolist(), Labels_pred_ensemble.tolist()), auc(
        Ensemble_recall, Ensemble_precision)
    fig = pyplot.figure()
    pyplot.plot(Ensemble_recall, Ensemble_precision, marker='.', label='SVM')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    fig.savefig('CNNs_bagging_pr_curve.png')
    pyplot.close()

    Ensemble_CM = confusion_matrix(Labels_test.tolist(), Labels_pred_ensemble.tolist())
    Ensemble_precision = precision_score(Labels_test.tolist(), Labels_pred_ensemble.tolist(), pos_label=1)
    Ensemble_recall = recall_score(y_true=Labels_test.tolist(), y_pred=Labels_pred_ensemble.tolist(),
                                   pos_label=1)

    print(" Ensembled %.4f    %.4f    %.4f    %.4f    %.4f" % (
    acc, Ensemble_precision, Ensemble_recall, Ensemble_auc, Ensemble_f1))

    Indicator = {
        "Labels_pred_all": Labels_pred,
        "Scores_pred_all": Scores_pred,
        "Precisions_all": Precisions_all,
        "Recall_all": Recall_all,
        "PR_AUC_all": PR_AUC_all,
        "F1_all": F1_all,
        "ACC_all": ACC_all,
        "Confusion_Matrix_all": Confusion_Matrix_all,
        "Labels_pred_ensemble": Labels_pred_ensemble,
        "Ensemble_CM": Ensemble_CM,
        "Ensemble_precision": Ensemble_precision,
        "Ensemble_recall": Ensemble_recall,
        "Ensemble_f1": Ensemble_f1,
        "Ensemble_auc": Ensemble_auc,
        "Ensemble_acc": acc,
        "Ensemble_probs": Ensemble_probs
    }
    joblib.dump(Indicator, './Images/Indicator_CNN.pkl')
    return Indicator

def analysis_Final(Data):
    current_path = os.path.dirname(__file__)
    os.chdir(current_path)
    ##### Multi-Modality Ensemble ####
    numclf = 2
    Indicator_transformers = joblib.load("./SMILES/Indicator.pkl")
    Indicator_RFs = joblib.load('./ECFP/Indicator.pkl')
    Indicator_SVMs = joblib.load('./Images/Indicator_SVM.pkl')
    Indicator_CNNs = joblib.load('./Images/Indicator_CNN.pkl')
    # Labels_final = Indicator_transformers["Labels_pred_ensemble"] + Indicator_RFs["Labels_pred_ensemble"] + Indicator_SVMs["Labels_pred_ensemble"] + Indicator_CNNs["Labels_pred_ensemble"]
    # Scores_final = Indicator_transformers["Ensemble_probs"] + Indicator_RFs["Ensemble_probs"] + Indicator_SVMs["Ensemble_probs"] + Indicator_CNNs["Ensemble_probs"]

    Labels_final =  Indicator_RFs["Labels_pred_ensemble"] + Indicator_SVMs["Labels_pred_ensemble"]
    Scores_final = Indicator_RFs["Ensemble_probs"] + Indicator_SVMs["Ensemble_probs"]
    Scores_final = Scores_final / numclf
    numsamples_test = len(Labels_final)

    for id_sample in range(numsamples_test):
        if Labels_final[id_sample] >= numclf/2.0:
            Labels_final[id_sample] = 1
        elif Labels_final[id_sample] <= numclf/2.0:
            Labels_final[id_sample] = 0
        else:
            if Scores_final[id_sample > 0.5]:
                Labels_final[id_sample] = 1
            else:
                Labels_final[id_sample] = 0


    Labels_test = Data['Labels_test']
    Final_precision, Final_recall, _ = precision_recall_curve(Labels_test.tolist(),
                                                              Scores_final[:, 1].tolist())
    Final_f1, Final_auc = f1_score(Labels_test.tolist(), Labels_final.tolist()), auc(
        Final_recall, Final_precision)
    fig = pyplot.figure()
    pyplot.plot(Final_recall, Final_precision, marker='.', label='SVM')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    fig.savefig('Final_pr_curve.png')
    pyplot.close()

    Final_CM = confusion_matrix(Labels_test.tolist(), Labels_final.tolist())
    Final_precision = precision_score(Labels_test.tolist(), Labels_final.tolist(), pos_label=1)
    Final_recall = recall_score(y_true=Labels_test.tolist(), y_pred=Labels_final.tolist(),
                                pos_label=1)
    Final_acc = accuracy_score(y_true=Labels_test.tolist(), y_pred=Labels_final.tolist())
    print(" Ensembled %.4f    %.4f    %.4f    %.4f    %.4f" % (
        Final_acc, Final_precision, Final_recall, Final_auc, Final_f1))
    Indicator = {
        "Labels_pred_ensemble": Labels_final,
        "Final_CM": Final_CM,
        "Final_precision": Final_precision,
        "Final_recall": Final_recall,
        "Final_f1": Final_f1,
        "Final_auc": Final_auc,
        "Final_acc": Final_acc,
        "Final_probs": Scores_final
    }
    joblib.dump(Indicator, 'Indicator_Final.pkl')
    return Indicator