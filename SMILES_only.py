import joblib
import numpy as np
import os as os
from torch import nn
from torch.nn import functional
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
import torch
####### 仅仅用SMILES作为特征设计分类器 ########

#### (1) 采用Transformer 网络直接由文本进行分类
#### 预训练模型为 Distilbert 文本分类模型
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset,Dataset
from transformers import DistilBertTokenizer
import datasets
from datasets import load_metric

def construct_dataset(Data):
    SMILES_org_train = Data['SMILES_org_train'].tolist()
    Labels_train = Data['Labels_train'].tolist()
    SMILES_org_val = Data['SMILES_org_val'].tolist()
    Labels_val = Data['Labels_val'].tolist()
    SMILES_org_test = Data['SMILES_org_test'].tolist()
    Labels_test = Data['Labels_test'].tolist()

    texts = [s[0] for s in SMILES_org_train]
    my_dict = {"label": Labels_train, "text": texts}
    dataset_train = Dataset.from_dict(my_dict)

    texts = [s[0] for s in SMILES_org_val]
    my_dict = {"label": Labels_val, "text": texts}
    dataset_val = Dataset.from_dict(my_dict)

    texts = [s[0] for s in SMILES_org_test]
    my_dict = {"label": Labels_test, "text": texts}
    dataset_test = Dataset.from_dict(my_dict)

    dataset_all = datasets.DatasetDict({"train": dataset_train, "test": dataset_test, "val": dataset_val})
    return dataset_all

def construct_test_dataset(Data,tokenizer):
    SMILES_org_test = Data['SMILES_org_test'].tolist()
    test_text = [s[0] for s in SMILES_org_test]
    pt_batch = tokenizer(
        test_text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    return pt_batch

def transformer_SMILES_origin(Data,output_dir="./SMILES/"):
    dataset_all=construct_dataset(Data)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def tokenize_function(examples):
        return tokenizer(examples["text"],padding='max_length',truncation=True)

    tokenized_datasets = dataset_all.map(tokenize_function,batched=True)
    print(tokenized_datasets)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
    )


    # metric = load_metric("accuracy")
    metric = load_metric("roc_auc")
    def compute_metric(eval_pred):
        logits,labels = eval_pred
        predictions = np.argmax(logits,axis=-1)
        return metric.compute(predictions = predictions,references = labels)

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset= tokenized_datasets['train'].shuffle(seed=42),
        eval_dataset=tokenized_datasets['test'].shuffle(seed=42),
        compute_metrics=compute_metric,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    model.cpu()
    torch.cuda.empty_cache()

    # trainer.evaluate()
    # save_directory = output_dir
    # tokenizer.save_pretrained(save_directory)
    # model.save_pretrained(save_directory)
    return model, tokenized_datasets,tokenizer

def transfomer_bagging(Data):
    X_train = Data['SMILES_org_train']
    Labels_train = Data['Labels_train']
    X_val = Data['SMILES_org_val']
    Labels_val = Data['Labels_val']
    X_test = Data['SMILES_org_test']
    Labels_test = Data['Labels_test']

    X_train_pos=X_train[Labels_train==1,:]
    X_train_neg = X_train[Labels_train == 0, :]
    Labels_train_pos = Labels_train[Labels_train==1]
    Labels_train_neg = Labels_train[Labels_train == 0]

    numpos=sum(Labels_train==1)
    numneg=sum(Labels_train==0)

    numclassifer=int(numneg/numpos)
    Models_bagging=[]
    for id_calssifer in range(numclassifer):
        torch.cuda.empty_cache()
        x_train=np.concatenate((X_train_pos,X_train_neg[id_calssifer*numpos:(id_calssifer+1)*numpos,:]),0)
        y_train=np.concatenate((Labels_train_pos,Labels_train_neg[id_calssifer*numpos:(id_calssifer+1)*numpos]),0)
        Data_mini={
            'SMILES_org_train':x_train,
            'Labels_train':y_train,
            'SMILES_org_val':X_val,
            'Labels_val':Labels_val,
            'SMILES_org_test':X_test,
            'Labels_test':Labels_test
        }
        model_transformer, tokenized_datasets, tokenizer = transformer_SMILES_origin(Data_mini,output_dir="./SMILES/Transformer/"+str(id_calssifer))
        Models_bagging.append(model_transformer)
    joblib.dump(Models_bagging, './SMILES/Transformers_bagging.pkl')
    joblib.dump(tokenizer, './SMILES/tokenizer.pkl')

    return Models_bagging, tokenizer

