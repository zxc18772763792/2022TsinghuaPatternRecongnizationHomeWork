from Analysis import analysis_transformers, analysis_RFs, analysis_SVMs, analysis_CNNs, analysis_Final, \
    Indicator_toExcel, Final_Indicator_toExcel
from ProcessData import processdata
from SMILES_only import transfomer_bagging
from ECFP_only import my_forests_ECFP
from Images_only import my_svm_Images
from CNN_Images import CNNs_bagging
import joblib
Data=processdata(total_ratio=0.1,train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)

###### Bagging Transformer models with SMILES as feature ########
Models_bagging, tokenizer=transfomer_bagging(Data)
Indicator_transformers=analysis_transformers(Data)
Indicator_toExcel(Indicator_transformers,'./SMILES/Indicator_Transformer.xlsx',Data['Labels_test'].tolist())
##### Bagging RandomForests models with ECFP as feature #######
numfeature_selected=512
RF_bagging,selected_feature_id=my_forests_ECFP(Data,numfeature_selected=numfeature_selected)
Indicator_RFs=analysis_RFs(Data)
Indicator_toExcel(Indicator_RFs,'./ECFP/Indicator_RF.xlsx',Data['Labels_test'].tolist())

##### PCA+SVM with Image as feature #####
pca_components=64
SVM_bagging,pca=my_svm_Images(Data,pca_components=pca_components)
Indicator_SVMs=analysis_SVMs(Data)
Indicator_toExcel(Indicator_SVMs,'./Images/Indicator_SVM.xlsx',Data['Labels_test'].tolist())

##### CNN for Images as feature #####
CNNs=CNNs_bagging(Data)
Indicator_CNNs=analysis_CNNs(Data)
Indicator_toExcel(Indicator_CNNs,'./Images/Indicator_CNN.xlsx',Data['Labels_test'].tolist())

##### Final Multi-Modality Models
Indicator_Final=analysis_Final(Data)
Final_Indicator_toExcel(Indicator_Final, 'Indicator_Final.xlsx',Data['Labels_test'].tolist())

