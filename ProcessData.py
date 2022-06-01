import numpy as np
import os

def processdata(total_ratio=0.1,train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    print('total_ratio='+str(total_ratio))
    print('train_ratio=' + str(train_ratio))
    print('val_ratio=' + str(val_ratio))
    print('test_ratio=' + str(test_ratio))
    current_path = os.path.dirname(__file__)
    os.chdir(current_path)
    try:
        os.mkdir('./SMILES')
    except:
        pass

    try:
        os.mkdir('./ECFP')
    except:
        pass

    try:
        os.mkdir('./Images')
    except:
        pass
    ##### 读取数据
    SMILES_origin=np.load('SMILES.npy').astype('str')
    ECFP=np.load('ECFP.npy')
    Images=np.load('Image.npy')
    Labels=np.load('label.npy')
    numMolecule=len(Labels)

    ####### 预处理特征和数据 #######

    ## 将SMILES特征转为长度固定的10进制的一维向量
    nums=[]
    for id_Molecule,SMILE in enumerate(SMILES_origin):
        nums.append(len(SMILE))
        maxnum=max(nums) #向量长度

    SMILES=np.zeros([numMolecule,maxnum])
    for id_Molecule,SMILE in enumerate(SMILES_origin):
        for id_s, s in enumerate(SMILE):
            SMILES[id_Molecule, id_s]=ord(s)

    ### 将数据分成7：1：2分别作为训练集、验证集和测试集
    ### 分为几个集合的时候保证正负样本比例不变
    numpos=sum(Labels==1)
    numneg=sum(Labels==0)

    SMILES_pos=SMILES[Labels==1,:]
    SMILES_neg=SMILES[Labels==0,:]

    SMILES_origin = SMILES_origin.reshape(numMolecule, 1)
    SMILES_org_pos = SMILES_origin[Labels == 1, :]
    SMILES_org_neg = SMILES_origin[Labels == 0, :]

    ECFP_pos = ECFP[Labels == 1, :]
    ECFP_neg = ECFP[Labels == 0, :]

    Images_pos = Images[Labels == 1, :]
    Images_neg = Images[Labels == 0, :]

    Labels_pos=Labels[Labels==1]
    Labels_neg=Labels[Labels==0]

    numpos=int(numpos*total_ratio)
    numneg=int(numneg * total_ratio)

    SMILES_pos=SMILES_pos[0:numpos,:]
    SMILES_neg = SMILES_neg[0:numneg,:]
    SMILES_train=np.concatenate((SMILES_pos[0:int(numpos*train_ratio),:],SMILES_neg[0:int(numneg*train_ratio),:]),0)
    SMILES_val=np.concatenate((SMILES_pos[int(numpos*train_ratio)+1:int(numpos*train_ratio)+int(numpos*val_ratio),:],SMILES_neg[int(numneg*train_ratio)+1:int(numneg*train_ratio)+int(numneg*val_ratio),:]),0)
    SMILES_test=np.concatenate((SMILES_pos[-int(numpos*test_ratio):-1,:],SMILES_neg[-int(numneg*test_ratio):-1,:]),0)


    SMILES_org_pos=SMILES_org_pos[0:numpos,:]
    SMILES_org_neg=SMILES_org_neg[0:numneg, :]
    SMILES_org_train=np.concatenate((SMILES_org_pos[0:int(numpos*train_ratio),:],SMILES_org_neg[0:int(numneg*train_ratio),:]),0)
    SMILES_org_val=np.concatenate((SMILES_org_pos[int(numpos*train_ratio)+1:int(numpos*train_ratio)+int(numpos*val_ratio),:],SMILES_org_neg[int(numneg*train_ratio)+1:int(numneg*train_ratio)+int(numneg*val_ratio),:]),0)
    SMILES_org_test=np.concatenate((SMILES_org_pos[-int(numpos*test_ratio):-1,:],SMILES_org_neg[-int(numneg*test_ratio):-1,:]),0)


    ECFP_pos=ECFP_pos[0:numpos,:]
    ECFP_neg=ECFP_neg[0:numneg, :]
    ECFP_train=np.concatenate((ECFP_pos[0:int(numpos*train_ratio),:],ECFP_neg[0:int(numneg*train_ratio),:]),0)
    ECFP_val=np.concatenate((ECFP_pos[int(numpos*train_ratio)+1:int(numpos*train_ratio)+int(numpos*val_ratio),:],ECFP_neg[int(numneg*train_ratio)+1:int(numneg*train_ratio)+int(numneg*val_ratio),:]),0)
    ECFP_test=np.concatenate((ECFP_pos[-int(numpos*test_ratio):-1,:],ECFP_neg[-int(numneg*test_ratio):-1,:]),0)


    Images_pos=Images_pos[0:numpos,:]
    Images_neg=Images_neg[0:numneg, :]
    Images_train=np.concatenate((Images_pos[0:int(numpos*train_ratio),:],Images_neg[0:int(numneg*train_ratio),:]),0)
    Images_val=np.concatenate((Images_pos[int(numpos*train_ratio)+1:int(numpos*train_ratio)+int(numpos*val_ratio),:],Images_neg[int(numneg*train_ratio)+1:int(numneg*train_ratio)+int(numneg*val_ratio),:]),0)
    Images_test=np.concatenate((Images_pos[-int(numpos*test_ratio):-1,:],Images_neg[-int(numneg*test_ratio):-1,:]),0)



    Labels_pos=Labels_pos[0:numpos]
    Labels_neg=Labels_neg[0:numneg]
    Labels_train=np.concatenate((Labels_pos[0:int(numpos*train_ratio)],Labels_neg[0:int(numneg*train_ratio)]),0)
    Labels_val=np.concatenate((Labels_pos[int(numpos*train_ratio)+1:int(numpos*train_ratio)+int(numpos*val_ratio)],Labels_neg[int(numneg*train_ratio)+1:int(numneg*train_ratio)+int(numneg*val_ratio)]),0)
    Labels_test=np.concatenate((Labels_pos[-int(numpos*test_ratio):-1],Labels_neg[-int(numneg*test_ratio):-1]),0)

    numMolecule=int(numpos*total_ratio)+int(numneg * total_ratio)
    Data={
        'numMolecule':numMolecule,
        'SMILES_org_train':SMILES_org_train,
        'SMILES_org_val':SMILES_org_val,
        'SMILES_org_test':SMILES_org_test,
        'SMILES_train':SMILES_train,
        'SMILES_val':SMILES_val,
        'SMILES_test':SMILES_test,
        'ECFP_train':ECFP_train,
        'ECFP_val':ECFP_val,
        'ECFP_test':ECFP_test,
        'Images_train':Images_train,
        'Images_val':Images_val,
        'Images_test':Images_test,
        'Labels_train':Labels_train,
        'Labels_val':Labels_val,
        'Labels_test':Labels_test,
    }

    return Data
