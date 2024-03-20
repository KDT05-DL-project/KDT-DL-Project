import numpy as np
import pandas as pd
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ClassModel(nn.Module):
    def __init__(
        self,
        in_: int,
        out_: int,
        node_list: list = [100],
        use_drop: bool = False,
        drop_ratio: float = 0.5,
        drop_freq: int = 1,
    ):
        super().__init__()
        node_list = [in_] + node_list + [out_]
        layer_num = len(node_list) - 1
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(nn.Linear(node_list[i], node_list[i + 1]))
            if i != layer_num - 1:
                if use_drop and (i % drop_freq) == 0:
                    self.layers.append(nn.Dropout(drop_ratio))
                self.layers.append(nn.BatchNorm1d(node_list[i + 1]))
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def load_data(path):
    model = torch.load(path + "best_model_no_drop_6.pth")
    X: pd.DataFrame = pd.read_pickle(path + "feature2.pkl")
    y: pd.DataFrame = pd.read_pickle(path + "target2.pkl")

    fish_list = [
        "Black Sea Sprat",
        "Gilt-Head Bream",
        "Hourse Mackerel",
        "Red Mullet",
        "Red Sea Bream",
        "Sea Bass",
        "Shrimp",
        "Striped Red Mullet",
        "Trout",
    ]
    index = [i for i in range(len(fish_list))]

    fish_dict = dict(zip(index, fish_list))  # 생선 이름과 인덱스 번호 매핑

    return model, X, y, fish_dict


def predict(model, X: pd.DataFrame, num: int):
    data = torch.FloatTensor(X.iloc[[num]].values)
    proba = F.softmax(model(data), dim=1).max().item()
    proba = round(proba * 100, 2)
    result = model(data).argmax().item()
    return proba, result


def show_fish(X: pd.DataFrame, num: int):
    plt.figure(figsize=(5, 4))
    plt.imshow(X.iloc[num].values.reshape(45, -1), cmap="binary")
    plt.axis("off")
    plt.show()


def displayUI():
    dir = "../DATA/"
    model, X, y, fish_dict = load_data(dir)
    while True:
        os.system("cls")
        print("===============================================")
        print("            생선 종류 예측 프로그램")
        print("===============================================")
        num = input("데이터 번호를 입력하세요(0~8999), -1로 종료:\n")
        if num == "-1":
            print("프로그램을 종료합니다.")
            break
        elif num.isdecimal() and (0 <= int(num) <= 8999):
            num = int(num)
            proba, result = predict(model, X, int(num))
            print(f"생선 종류: {fish_dict[result]} (정확도: {proba}%)")
            show_fish(X, num)
            input("\n계속하려면 아무 키나 누르세요.")
        else:
            input("잘못된 입력입니다. 다시 입력하세요.")


def main():
    displayUI()


if __name__ == "__main__":
    main()
