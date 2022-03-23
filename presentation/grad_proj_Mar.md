---
marp: true
theme: my-theme
paginate: true
---
# 졸업프로젝트 중간점검
### 3월 과제: Keras 연습
응용미술교육과 
2018030328 신영은

---
# 목차
1. 신경망의 구조
2. **MLP**
    1. 이진 분류 문제: IMDB 데이터셋
    2. 다중 분류 문제: 로이터 데이터셋
    3. 회귀 문제: 보스턴 주택 데이터셋
3. **모델 튜닝**
    1. 과대적합
    2. k-fold 교차 검증
    3. 네트워크 용량 감소
    4. 가중치 규제
    5. 드롭 아웃
4. **CNN**
    1. 작은 데이터셋 이진 분류 문제: Kaggle Cats vs Dog
    2. 사전 훈련된 Convnet 사용하기: VGG16
    3. Convnet 학습 시각화하기
---
# 1. 신경망의 구조
![width:800px](https://blog.kakaocdn.net/dn/btlekM/btqzhhoqubL/klqNFRPkBWSakQMZrNd4Bk/img.png)
신경망의 구성 요소간의 관계

---
# 1. 신경망의 구조
## 1.1. 층
* 하나 이상의 텐서를 입력받아 하나 이상의 텐서를 출력하는 데이터 처리 모듈
* 대부분의 층은 **가중치**라는 상태를 가짐

### 1.1.1. 가중치:
* **확률적 경사 하강법**에 의해 학습
* 훈련 데이터를 신경망에 노출시켜 학습된 정보를 가짐
* 피드백 신호에 기초해 점진적으로 조정됨
---
### 1.1.2. 활성화 함수
* 이전 층의 결과값을 변환하여 다른 층으로 신호를 전달하는 역할
![width:700px](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbeswMt%2FbtqYpV2m5DU%2Fvqv8wX4oRhlM99eqhQIRx0%2Fimg.png)
---
# 1. 신경망의 구조
### 1.1.2. 층의 종류
#### 1.1.2.1 완전 연결 층 (fully connected layer)
* 2D 텐서가 저장된 간단한 벡터 데이터 처리
* keras에서는 ```Dense``` class
#### 1.1.2.2. 순환 층 (recurrent layer)
* 3D 텐서로 저장된 시퀀스 데이터 처리
* ```simple RNN```, ```LSTM``` 등
#### 1.1.2.3. 2D 합성곱 층 (convolution layer)
* 4D 텐서로 저장된 이미지 데이터 처리
* ```conv2D``` class
---
# 1. 신경망의 구조
## 1.2. 모델
* 층으로 만든 DAG(Directed Acyclic Graph)
* 네트워크 구조는 **가설 공간**을 정의함
    * 가설 공간(space of hypotheses)
        * 어떤 문제를 해결하는데 필요한 가능성 있는 가설 후보군의 집합
    * 네트워크 구조를 선택함으로써 가설 공간을 입력 데이터에서 출력 데이터로 매핑하는 일련의 특정 텐서 연산으로 제한
---
# 1. 신경망의 구조
## 1.3. 손실 함수
* 예측값과 실제값의 차이를 계산 -> **훈련하는 동안 최소화되야함**
### 1.3.1. 문제 유형에 맞는 손실 함수
|문제 유형|마지막 층의 활성화 함수|손실 함수|
|------|---|---|
|이진 분류|sigmoid|binary_crossentropy|
|단일 레이블 다중 분류|softmax|categorical_crossentropy|
|다중 레이블 다중 분류|sigmoid|binary_crossentropy|
|임의 값에 대한 회귀|-|mse|
|0과 1 사이 값에 대한 회귀|sigmoid|mse or binary_crossentropy|
## 1.4. 옵티마이저
* 손실 함수를 기반으로 네트워크가 어떻게 업데이트될지 결정
* 특정 종류의 확률적 경사 하강법을 구현
---
# 2. MLP 예제
## 2.1. 이진 분류 문제
> IMDB dataset 영화 리뷰 분류
### 2.1.1. 데이터 준비하기
#### 2.1.1.1 IMDB 데이터셋 로드 후 훈련, 테스트 데이터로 나누기
* 훈련과 테스트를 같은 데이터로 하면 안됨!
    * 모델은 처음 보는 데이터에 대한 성능이 중요!
    * 모델에는 이미 훈련 데이터에 맞는 규칙이 반영되었기 때문에 성능 평가 단계에서는 학습에 사용되지 않은 데이터 사용

```python
from keras.datasets import imdb
# 가장 자주 사용되는 단어 10,000개만 사용
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
```
---
## 2.1. 이진 분류 문제
### 2.1.1.2. 신경망에 주입할 데이터 준비하기
* 숫자 리스트를 원-핫 인코딩하여 0과 1의 벡터로 변환 
* **원-핫 인코딩**:
    1. 단어 집합의 크기를 벡터의 차원 만들기 
    2. 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스는 모두 0으로 만들기

```python
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
```
---
## 2.1. 이진 분류 문제
### 2.1.2. 신경망 모델 만들기
#### 2.1.2.1 완전 연결 신경망
* ```relu``` 활성화 함수를 사용한 ```Dense```층 쌓기
    1. 16개의 은닉 유닛을 가진 두 개의 은닉층
    2. 현재 리뷰의 감정을 스칼라 값의 예측으로 출력하는 마지막 층
        * ```sigmoid```사용: 임의의 값을 [0, 1] 사이로 압축 -> 출력 값을 **확률**처럼 해석
```python
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```
---
## 2.1. 이진 분류 문제
#### 2.1.2.2. 옵티마이저, 손실함수 설정
* 옵티마이저: rmsprop
* 손실함수: 
    * 이진 분류 -> binary_crossentropy
    * 확률을 출력하는 모델이므로 확률 분포 간의 차이를 측정하는 크로스엔트로피 쓰기 (원본 분포와 예측 분포 사이를 측정)
```python
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy']) # 정확도를 사용해 모니터링
```
---
## 2.1. 이진 분류 문제
### 2.1.3. 훈련 & 검증하기
#### 2.1.3.1. 검증 세트 만들기
* 훈련 데이터에서 일부 데이터 떼어내 검증 세트 만들기
* 검증용 데이터는 훈련에서 사용되면 안됨! (처음 본 데이터에 대한 모델의 정확도 측정)
#### 2.1.3.2. 모델 훈련하기
* ```model.fit()```: 
    * ```History``` 객체 반환
    * 훈련하면서 발생한 모든 정보를 담고있는 딕셔너리  ```history```속성을 가짐
```python
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
```
---
## 2.1. 이진 분류 문제
#### 2.1.3.3. 훈련과 검증 그래프
* matplot으로 훈련과 검증 손실 & 정확도 그래프 그리기
![width:850px](./image/binary_graph.png)
* 4번째 epoch부터 **과대적합** 발생!
    * 훈련 데이터에 과도하게 최적화되어 새로운 데이터에 일반화 X
    * 과대적합 발생 전까지만 훈련하고 테스트데이터로 평가하기
---
## 2.1. 이진 분류 문제
### 2.1.4. 테스트 데이터에서 평가
* 과대적합 발생 전까지만 훈련한 후 test 데이터로 평가하기
* ```evaluate()```: 모델의 최종적인 정답률과 loss값 알 수 있음
```python 
results = model.evaluate(x_test, y_test)
```
### 2.1.5. 훈련된 모델로 새로운 데이터에 대해 예측하기
* ```predict()```: 데이터가 양성 샘플(label=1)일 확률 예측
```python
model.predict(x_test)
```
### 2.1.6. 정리
1. 원본 데이터를 신경망에 주입 전에 **전처리** 하기
1. 출력 class가 2개인 **이진분류** 문제는 **1개의 unit**과 **Sigmoid** 활성화 함수를 가진 Dense 층으로 끝내기
2. 이진 분류의 스칼라 sigmoid 출력에는 **binary_crossentropy** 손실함수 쓰기
---
## 2.2. 다중 분류 문제
> 로이터 dataset 뉴스 토픽(46가지) 분류
