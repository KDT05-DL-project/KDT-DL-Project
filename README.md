# _KDT05-Machine Learning Project_

경북대학교 KDT(Korea Digital Training) 빅데이터 전문가 양성과정 5기 : DL(Deep Learning) 3팀입니다

명노아 : [깃허브 링크](https://github.com/noah2397)  
이윤서 : [깃허브 링크](https://github.com/voo0o08)  
김동현 : [깃허브 링크](https://github.com/DongHyunKKK)  
이현길 : [깃허브 링크](https://github.com/Schubert3275)

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

<hr/>

#### 개발환경

| 패키지 이름 | 버전   |
| ----------- | ------ |
| Python      | 3.8.18 |
| torch       | 2.2.0  |
| pandas      | 2.0.3  |
| matplotlib  | 3.7.2  |
| sns         | 0.12.2 |
| np          | 1.24.3 |
| sklearn     | 1.3.0  |

<hr/>

### KDT(Korea Digital Training)-DL(Deep learning)

<hr/>

#### 사용한 데이터 사이트(수정 전)

1. [Kaggle](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset)

<hr/>

###### 주제 : 딥러닝의 딥러닝

- 목차

* 1. 역할 분담
* 2. 주제 선정 배경
* 3. 데이터 선정 배경
* 4. 소주제 개요
* 5. 옵티마이저와 성능의 상관관계(명노아)
* 6. 퍼셉트론과 성능의 상관관계(이윤서)
* 7. 은닉층과 성능의 상관관계(김동현)
* 8. 최종 모델 구현(이현길)
* 9. 예측 서비스 구현
* 10. 출처
  </hr>

###### 역할 분담

|          역할 | 김동현 | 이윤서 | 명노아 | 이현길 |
| ------------: | ------ | ------ | ------ | ------ |
|      주제선정 | ✅     |        |        | ✅     |
| 데이터셋 선정 |        |        | ✅     | ✅     |
| 이미지 전처리 |        | ✅     | ✅     |        |
|        Readme | ✅     | ✅     | ✅     | ✅     |
|   서비스 구현 | ✅     | ✅     |        | ✅     |

### 소주제 개요(개인 항목)

<details>
  <summary>
    명노아
  </summary>
</details>

</hr>

<details>
  <summary>
    이윤서(퍼셉트론 수와 성능의 관계)
  </summary>

- 실험 1 : 히든 레이어 없는 상태로 퍼셉트론 수 변경
    
    <div class="aside">
    
    ✏️ perceptron_num → 100
    ```
    Linear(in_features=2655, out_features=perceptron_num, bias=True)
    ReLU()
    Linear(in_features=perceptron_num, out_features=9, bias=True)
    ```
    </div>
    
    **결과**
    
    심하게 작은 값들은 n회 수행에도 score가 낮았지만, 일정 퍼셉트론 값 이상부터는 70 정도로 결과가 나오는 것을 확인할 수 있었다.
    
    <img width="419" alt="스크린샷 2024-03-20 115546" src="https://github.com/voo0o08/KDT-DL-Project/assets/155411941/f4f13d7e-7ca9-482d-8579-7bc500acceb0">
    
- 실험 2 : 히든 레이어 2개인 상태로 퍼셉트론 수 변경1
    
    <div class="aside">
    ✏️ perceptron_num → 100
      
    ```
    Linear(in_features=2655, out_features=perceptron_num, bias=True)
    ReLU()
    Linear(in_features=perceptron_num, out_features=perceptron_num, bias=True)
    ReLU()
    Linear(in_features=perceptron_num, out_features=perceptron_num, bias=True)
    ReLU()
    Linear(in_features=perceptron_num, out_features=9, bias=True)
    ```
    
    </div>
    
    **결과**
    
    마찬가지로 값이 심하게 크거나 작지 않은 이상 값이 70 정도임을 알 수 있었고, 1번 실험은 전체 수행 시간이 10분인 반면 레이어가 생기고 퍼셉트론 수를 높이니 계산 비용이 증가하여 1시간 이상 실행되었다. 또한 들인 시간에 비해 score가 높아지지 않았다. 
    
    
    <img width="422" alt="스크린샷 2024-03-20 120126" src="https://github.com/voo0o08/KDT-DL-Project/assets/155411941/8c642c77-81fa-4f98-86df-c4b68b5a2d3c">


    
- 실험 3 : 히든 레이어 2개인 상태로 퍼셉트론 수 변경2
    
    <aside>
    ✏️ perceptron_num → n1, n2, n3…
      
    ```
    Linear(in_features=2655, out_features=n1, bias=True)
    ReLU()
    Linear(in_features=n1, out_features=n2, bias=True)
    ReLU()
    Linear(in_features=n2, out_features=n3, bias=True)
    ReLU()
    Linear(in_features=n3, out_features=9, bias=True)
    ```
    
    </aside>
    
    **결과**
    
    2번 실험과 달리 레이어수를 출력층에 가까울 수록 작아지게 설정하였다. 수행 시간 15분으로 줄었고, 결과에는 큰 차이가 없음을 알 수 있었다.
    
    <img width="318" alt="스크린샷 2024-03-20 155147" src="https://github.com/voo0o08/KDT-DL-Project/assets/155411941/c34741d8-2b2f-4dcf-b617-5d8403b52eff">


**결론**

퍼셉트론은 경우의 수가 너무 많은 하이퍼파라미터로 ‘특정 값을 찾겠다!’ 보다는 ‘이 정도 값은 제외해야겠다’ 정도로 사용하고 다른 하이퍼파라미터를 조정하는 것이 효과적이다.
</details>

</hr>

<details>
  <summary>
    김동현
  </summary>

</details>

</hr>

<details>
  <summary>
    이현길
  </summary>

</details>
<hr/>
