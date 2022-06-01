import torch
import torch.utils.data as data
from CNN_Images import myDataset
from SMILES_only import transfomer_bagging, construct_test_dataset
import joblib
import numpy as np
from torch import nn
from torch.nn import functional
from sklearn.preprocessing import StandardScaler
import os
import sys

def test_ensemble_transformer_single_sample(Data):
    ###### SMILES Only ##########
    # Single Transformer Classifer
    Models_bagging=joblib.load('./SMILES/Transformers_bagging.pkl')
    tokenizer=joblib.load('./SMILES/tokenizer.pkl')

    numclassifer=len(Models_bagging)
    pt_batch = tokenizer(
        Data['SMILES_org_test'],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # pt_batch=construct_test_dataset(Data, tokenizer)
    Labels_pred = np.zeros(numclassifer)
    Scores_pred=np.zeros([numclassifer,2])

    for id_classifer, classifer in enumerate(Models_bagging):
        with torch.no_grad():
            classifer.eval()
            output = classifer(**pt_batch)
            pt_predictions = nn.functional.softmax(output.logits, dim=-1)
            Scores_pred[id_classifer,:]=pt_predictions.detach().numpy()
            pred = Scores_pred
            if pred[0,0] >= pred[0,1]:
                Labels_pred[id_classifer] = 0
            else:
                Labels_pred[id_classifer] = 1
    Labels_pred_ensemble=Labels_pred.sum(axis=0)
    if Labels_pred_ensemble >= numclassifer / 2:
        Labels_pred_ensemble = 1
    else:
        Labels_pred_ensemble = 0

    Ensemble_probs = Scores_pred.sum(axis=0) / numclassifer
    return Labels_pred_ensemble,Ensemble_probs

def test_ensemble_RF_single_sample(Data):
    ###### ECFP Only ##########
    # Single RF Classifer
    RFs_bagging = joblib.load('./ECFP/RF_bagging.pkl')
    selected_feature_id = joblib.load('./ECFP/selected_feature_id.pkl')

    X_test = Data['ECFP_test']
    numclassifer = len(RFs_bagging)

    Labels_pred = np.zeros(numclassifer)
    Scores_pred = np.zeros([numclassifer, 2])
    for id_classifer, classifer in enumerate(RFs_bagging):
        Scores_pred[id_classifer, :] = classifer.predict_proba(X_test[selected_feature_id[:, id_classifer]].reshape(1,-1))
        Labels_pred[id_classifer] = classifer.predict(X_test[selected_feature_id[:, id_classifer]].reshape(1,-1))

    Labels_pred_ensemble = Labels_pred.sum(axis=0)
    if Labels_pred_ensemble >= numclassifer / 2:
        Labels_pred_ensemble = 1
    else:
        Labels_pred_ensemble = 0

    Ensemble_probs = Scores_pred.sum(axis=0) / numclassifer
    return Labels_pred_ensemble,Ensemble_probs

def test_ensemble_SVM_single_sample(Data):
    ###### Images Only ##########
    # Single SVM Classifer
    SVMs_bagging = joblib.load('./Images/SVM_bagging.pkl')
    pca = joblib.load('./Images/pca.pkl')
    X_test = Data['Images_test']
    X_test = StandardScaler().fit_transform(X_test.reshape(1,-1))
    X_test = pca.transform(X_test)

    numclassifer = len(SVMs_bagging)

    Labels_pred = np.zeros(numclassifer)
    Scores_pred = np.zeros([numclassifer, 2])
    for id_classifer, classifer in enumerate(SVMs_bagging):
        Scores_pred[id_classifer, :] = classifer.predict_proba(X_test.reshape(1,-1))
        Labels_pred[id_classifer] = classifer.predict(X_test.reshape(1,-1))
    Labels_pred_ensemble = Labels_pred.sum(axis=0)
    if Labels_pred_ensemble >= numclassifer / 2:
        Labels_pred_ensemble = 1
    else:
        Labels_pred_ensemble = 0

    Ensemble_probs = Scores_pred.sum(axis=0) / numclassifer
    return Labels_pred_ensemble,Ensemble_probs

def test_single_sample(Data):
    Label_TR, Probs_TR= test_ensemble_transformer_single_sample(Data)
    Label_RF, Probs_RF = test_ensemble_RF_single_sample(Data)
    Label_SVM, Probs_SVM = test_ensemble_SVM_single_sample(Data)

    Probs_Final=(Probs_TR+Probs_RF+Probs_SVM)/3.0
    if Probs_Final[1]>=0.5:
        Label_Final = 1
    else:
        Label_Final = 0
    print(" True Label of test sample  is " + str(Data['Labels_test']))
    print(" Prediction Label of test sample  is "+str(Label_Final))
    print(" Prediction Score of test sample being positive is "+str(Probs_Final[1]))

def test_single_transformer_single_sample(Data):
    classifer=joblib.load('./SMILES/Best_TR.pkl')
    tokenizer = joblib.load('./SMILES/tokenizer.pkl')
    pt_batch = tokenizer(
        Data['SMILES_org_test'],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        classifer.eval()
        output = classifer(**pt_batch)
        pt_predictions = nn.functional.softmax(output.logits, dim=-1)
        Scores_pred = pt_predictions.detach().numpy()
        pred = Scores_pred
        if pred[0, 0] >= pred[0, 1]:
            Label_pred = 0
        else:
            Label_pred = 1
    print(" True Label of test sample  is " + str(Data['Labels_test']))
    print(" Single DistilBert Prediction Label of test sample  is " + str(Label_pred))
    print(" Single DistilBert Prediction Score of test sample being positive is " + str(Scores_pred[0,1]))

def test_single_RF_single_sample(Data):
    X_test = Data['ECFP_test']
    classifer=joblib.load('./ECFP/Best_RF.pkl')
    selected_feature_id=joblib.load('./ECFP/Best_feature.pkl')
    Scores_pred = classifer.predict_proba(X_test[selected_feature_id].reshape(1, -1))
    Label_pred = classifer.predict(X_test[selected_feature_id].reshape(1, -1))
    print(" True Label of test sample  is " + str(Data['Labels_test']))
    print(" Single Random Forest Prediction Label of test sample  is " + str(Label_pred[0]))
    print(" Single Random Forest Prediction Score of test sample being positive is " + str(Scores_pred[0,1]))

def test_single_SVM_single_sample(Data):
    classifer=joblib.load('./Images/Best_SVM.pkl')
    pca = joblib.load('./Images/pca.pkl')
    X_test = Data['Images_test']
    X_test = StandardScaler().fit_transform(X_test.reshape(1, -1))
    X_test = pca.transform(X_test)
    Scores_pred = classifer.predict_proba(X_test.reshape(1, -1))
    Label_pred = classifer.predict(X_test.reshape(1, -1))
    print(" True Label of test sample  is " + str(Data['Labels_test']))
    print(" Single SVM Prediction Label of test sample  is " + str(Label_pred[0]))
    print(" Single SVM Prediction Score of test sample being positive is " + str(Scores_pred[0,1]))

def test_single_CNN_single_sample(Data):
    dataset_test = myDataset(Data['Images_test'].reshape(1,-1), np.array(float(Data['Labels_test'])).reshape(-1))
    test_loader = data.DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, num_workers=0)
    classifer=joblib.load('./Images/Best_CNN.pkl')
    classifer.eval()
    with torch.no_grad():
        for _, (test_img, test_label) in enumerate(test_loader):
            test_output = classifer(test_img)
            Scores_pred = test_output.detach().numpy()
            Label_pred = torch.max(test_output, 1)[1].data.squeeze().detach().numpy()
    print(" True Label of test sample  is " + str(Data['Labels_test']))
    print(" Single CNN Prediction Label of test sample  is " + str(Label_pred))
    print(" Single CNN Prediction Score of test sample being positive is " + str(Scores_pred[0,1]))

### 用最佳模型在测试集上测试
# Data=processdata(total_ratio=0.1,train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
# test_single_sample(Data)

Id_test=sys.argv[1]
Id_test=int(Id_test)
print(sys.argv[1])
current_path = os.path.dirname(__file__)
os.chdir(current_path)
SMILES_origin = np.load('SMILES.npy').astype('str')
ECFP = np.load('ECFP.npy')
Images = np.load('Image.npy')
Labels = np.load('label.npy')

Data={
        'SMILES_org_test':SMILES_origin[Id_test],
        'ECFP_test':ECFP[Id_test],
        'Images_test':Images[Id_test],
        'Labels_test':Labels[Id_test],
    }
test_single_transformer_single_sample(Data)
test_single_RF_single_sample(Data)
test_single_SVM_single_sample(Data)
test_single_CNN_single_sample(Data)