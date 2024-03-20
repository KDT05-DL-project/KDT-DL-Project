

def optim_func(opt_name, opt):


    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    # https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset


    feature = pd.read_pickle("feature.pkl")
    target = pd.read_pickle("target.pkl")
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.1, random_state=42, stratify=target)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.1, random_state=42, stratify=y_test)
    x_train = torch.tensor(x_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    x_valid = torch.tensor(x_valid.values, dtype=torch.float32)
    y_valid = torch.tensor(y_valid.values, dtype=torch.long)
    x_test = torch.tensor(x_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.long)
    # 데이터셋 생성
    import torch.utils.data.dataloader as dataloader
    import torch.utils.data.dataset as dataset

    class MyDataset(dataset.Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y
        def __len__(self):
            return len(self.x)
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]
    dataset_train = MyDataset(x_train, y_train)
    dataset_test = MyDataset(x_test, y_test)
    dataset_valid = MyDataset(x_valid, y_valid)
    class model(nn.Module): # 가중치 초기화 
        def __init__(self):
            super(model, self).__init__()
            self.fc1 = nn.Linear(15000, 4000)
            self.fc2 = nn.Linear(4000, 9)
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    model2 = model()
    optimizer = opt(model2.parameters())
    criterion = nn.CrossEntropyLoss()

    train_loader = dataloader.DataLoader(dataset_train, batch_size=100, shuffle=True, drop_last=True)
    test_loader = dataloader.DataLoader(dataset_test, batch_size=100, shuffle=True, drop_last=True)
    valid_loader = dataloader.DataLoader(dataset_valid, batch_size=100, shuffle=True, drop_last=True)
    from torchmetrics.functional.classification import accuracy, f1_score
    model2.train()
    valid_score=[]
    train_score=[]

    for epoch in range(1):
        
        for x,y in train_loader:
            output = model2(x)
            loss = criterion(output, y.squeeze())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():        
            print(f"================================={epoch}=================================")
            output = model2(x_train)
            acc = accuracy(output, y_train.squeeze(), task="multiclass", num_classes=9, average="macro")
            f1 = f1_score(output, y_train.squeeze(), task="multiclass", num_classes=9, average="macro")
            train_score.append([acc, f1])
            print(f"train => acc : {acc}, f1 : {f1}")
            
            
            output = model2(x_valid)
            print(f"=================================valid score=================================")
            acc2 = accuracy(output, y_valid.squeeze(), task="multiclass", num_classes=9, average="macro")
            f12 = f1_score(output, y_valid.squeeze(), task="multiclass", num_classes=9, average="macro")
            valid_score.append([acc2, f12])
            print(f"valid => acc : {acc2}, f1 : {f12}")
    torch.save(model2,f"./model/{opt_name}_model.pth")
    output = model2(x_test)
    print(f"=================================test score=================================")
    print(accuracy(output, y_test.squeeze(), task="multiclass", num_classes=9, average="macro"))
    print(f1_score(output, y_test.squeeze(), task="multiclass", num_classes=9, average="macro"))
    return train_score, valid_score