<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    extensions: ["tex2jax.js"],
    jax: ["input/TeX", "output/HTML-CSS"],
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
      processEscapes: true
    },
    "HTML-CSS": { availableFonts: ["TeX"] }
  });
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Linear Regression 
작성자 : 전성욱
작성일 : 2017/03/28
수정일 : 2017/03/30
email : codetree@gmail.com

---

## 텐서플로 첫걸음 2장 - Linear Regression 

### 변수 간의 관계에 대한 모델
y = 0.1 * x + 0.3

가상의 학습데이타 만들기
~~~
num_points = 1000
vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

draw_plot_input_data(self.x_data, self.y_data)
~~~
![](images/make_input_data.png?raw=true)

### Hypothesis(가설)
\\( y = Wx + b \\)
~~~
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
~~~

### 비용함수와 경사 하강법 알고리즘
~~~
# 거리에 제곱을 하고 그합계에 대한 평균을 낸다.
loss = tf.reduce_mean(tf.square(y - y_data))
# 오차함수를 최소화하며 데이터에 가장 잘 들어 맞는 모델 찾기
# 경사 하강법 (Gradient Descent)
optimizer = tf.train.GradientDescentOptimizer(0.5)
# training
train = self.optimizer.minimize(loss)
~~~

### 알고리즘 실행 
초기화
~~~
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
~~~

학습
~~~
for step in range(learning_count):
    sess.run(self.train)

    print("step:{0} W:{1} b:{2}".format(step, sess.run(self.W), sess.run(self.b)))
    print("step:{0} loss:{1}".format(step, sess.run(self.loss)))
~~~

그리기
~~~
# 그래픽 표시
plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.xlabel('x')
plt.xlim(-2, 2)
plt.ylabel('y')
plt.ylim(0.1, 0.6)
plt.show()
~~~

### 전체 프로그램 
~~~
from __future__ import print_function

__author__ = 'codetree@google.com (sungwook Jeon)'

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

class class_linear_regression:
    """Linear Regression CLASS"""

    # Test Data #####################################################
    x_data = []
    y_data = []

    # .... Data #####################################################

    def __init__(self, name):
        self.name = name

    # 입력 Data를 그림으로 표현
    def draw_plot_input_data(self, x_data, y_data):
        plt.plot(x_data, y_data, 'ro')
        plt.show()

    # 입력 Data생성
    def input_data(self):
        num_points = 1000
        vectors_set = []

        for i in range(num_points):
            x1 = np.random.normal(0.0, 0.55)
            y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
            vectors_set.append([x1, y1])

        self.x_data = [v[0] for v in vectors_set]
        self.y_data = [v[1] for v in vectors_set]

        self.draw_plot_input_data(self.x_data, self.y_data)

    # 변수정의: (추론)
    def def_variables(self): # ? inference
        self.W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
        self.b = tf.Variable(tf.zeros([1]))
        self.y = self.W * self.x_data + self.b

    # 비용함수 정의 : 평균제곱오차
    def def_cost_function(self):
        # 거리에 제곱을 하고 그합계에 대한 평균을 낸다.
        self.loss = tf.reduce_mean(tf.square(self.y - self.y_data))
        # 오차함수를 최소화하며 데이터에 가장 잘 들어 맞는 모델 찾기
        # 경사 하강법 (Gradient Descent)
        self.optimizer = tf.train.GradientDescentOptimizer(0.5)
        # training
        self.train = self.optimizer.minimize(self.loss)

    def draw_plot_training(self, sess, x_data, y_data, W, b):
        # 그래픽 표시
        plt.plot(x_data, y_data, 'ro')
        plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
        plt.xlabel('x')
        plt.xlim(-2, 2)
        plt.ylabel('y')
        plt.ylim(0.1, 0.6)
        plt.show()

    # 학습
    def training(self, sess, learning_count):
        for step in range(learning_count):
            sess.run(self.train)

            print("step:{0} W:{1} b:{2}".format(step, sess.run(self.W), sess.run(self.b)))
            print("step:{0} loss:{1}".format(step, sess.run(self.loss)))

            self.draw_plot_training(sess, self.x_data, self.y_data, self.W, self.b)

    def run(self, learning_count):
        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        lr.training(sess, learning_count)  # 학습

    def evaluation(self):
        # evaluation
        print("evaluation")


if __name__ == '__main__':
    argc = len(sys.argv)
    if argc >= 2:
        print(sys.argv)
        # run_main(01_basic-operations.kr.srt)
        lr = class_linear_regression(sys.argv[1])
    else:
        lr = class_linear_regression("Linear Regressio")

    lr.input_data() # 초기 입력 Data생성
    lr.def_variables() # 변수정의
    lr.def_cost_function() # 비용함수 정의 : 평균제곱오차
    lr.run(8)  # 실행
~~~


---

## 그럼 하나하나 다시 봅시다....초보 입장에서.....
김성은님 강의 내용을 기반으로 설명합니다.
(해당 예제와 소스는 Sung Kim님의 강의 자료입니다.)
## Predicting exam score(시험 성적 예측) : regression(회기)

| x(hours) | y(score) |
|:--------:|:--------:|
|    10    |    90    |
|     9    |    80    |
|     3    |    50    |
|     2    |    30    |

- 학습 : 학습시간(x)과 성적(y) &rarr;  training습(학습) &rarr; regression모델
- 추론 : 학습시간(x) &rarr;  regression모델 &rarr; 예측성적(y)

## 문제 푸는 방법

### Hypothesis(가설)을 세운다.
Linear Regression이라는 학습 내용에 가장 적합한 선(수식,방정식)을 찾는 것
$$H(x) = Wx + b$$

###  Cost function : 가설과 실제 data와 차이계산
어떤 수식이 적합한지 찾기위한 평가방법 선택 
![](images/cost_graph.png?raw=true)
$$H(x)-y$$
 &rarr; 메롱이다.
$${ (H(x)-y) }^{ 2 }$$
&rarr; 차이를 음수 제거, 차이가 클수록 불이익을 준다.

### Cost function : mean square(평균제곱오차)
$$ \frac { { (H({ x }^{ (1) }) - { y }^{ (1) }) }^{ 2 } + { (H({ x }^{ (2) }) - { y }^{ (2) }) }^{ 2 } + { (H({ x }^{ (3) }) - { y }^{ (3) }) }^{ 2 } }{ 3 } $$

$$ cost=\frac { 1 }{ m } \sum _{ i=1 }^{ m }{ { (H({ x }^{ (i) })-{ y }^{ (i) }) }^{ 2 } } $$
$$ H(x)=Wx+b $$

$$ cost(W,b)=\frac { 1 }{ m } \sum _{ i=1 }^{ m }{ { (H({ x }^{ (i) })-{ y }^{ (i) }) }^{ 2 } } $$

Goal: Minimize cost
	-  minimize cost(W, b) : cost가 가장 작은 W,b를 구하는 것이 Linear Regression의 training이다.

### 정리 : hypothesis & cost function

Hypothesis(가설)
$$ H(x) = Wx + b $$

Cost function(비용함수)
$$ cost(W,b)=\frac { 1 }{ m } \sum _{ i=1 }^{ m }{ { (H({ x }^{ (i) })-{ y }^{ (i) }) }^{ 2 } } $$

### Sample code

~~~
from __future__ import print_function
import tensorflow as tf

# traning data
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Try to find values for W and b that compute y_data = W * x_data + b
# (we know that W should be 1 and b 0, bunt Tensorflow will
# figure that out for us.)
# Variable로 지정해야 model이 update가능
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# Our hypothesis(가설)
hypothesis = W * x_data + b

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# minimize
a = tf.Variable(0.1) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initialize the variables. 
# we whill 'run' this first.
init = tf.global_variables_initializer()

# Launch the ghaph
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

~~~

### Placeholder
모델을 재활용 할수 있게 해준다.

~~~
from __future__ import print_function

import tensorflow as tf

# traning data
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Try to find values for W and b that compute y_data = W * x_data + b
# (we know that W should be 1 and b 0, bunt Tensorflow will
# figure that out for us.)
# Variable로 지정해야 model이 update가능
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# 저장공간 확보
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis(가설)
hypothesis = W * X + b

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
a = tf.Variable(0.1) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initialize the variables.
# we whill 'run' this first.
init = tf.global_variables_initializer()

# Launch the ghaph
sess = tf.Session()
sess.run(init) # 초기화

# Fit the line.
for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data}) # 실행시 값을 넣는다.
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b)) # 실행시 값을 넣는다.

# placeholder를 사용하여서 다른 input_data를 이용하여 재활용 가능한다
print(sess.run(hypothesis, feed_dict={X: 5})) # 결과 예측
print(sess.run(hypothesis, feed_dict={X: 2.5})) # 결과 예측
~~~

# Cost minimize cost algorithm 원리
## Simplified hypothesis
$$ H(x) = Wx $$
$$ cost(W)=\frac { 1 }{ m } \sum _{ i=1 }^{ m }{ { ({ W }^{ (i) }-{ y }^{ (i) }) }^{ 2 } } $$

## What cast(W) looks like?
$$ cost(W)=\frac { 1 }{ m } \sum _{ i=1 }^{ m }{ { ({ W }^{ (i) }-{ y }^{ (i) }) }^{ 2 } } $$

|     x    |     y    |
|:--------:|:--------:|
|     1    |     1    |
|     2    |     2    |
|     3    |     3    |

- W=1, cost(W)=0
$$ \frac { { (1\times 1-1) }^{ 2 }\quad +\quad { (1\times 2-2) }^{ 2 }\quad +\quad { (1\times 3-3) }^{ 2 } }{ 3 } \quad =\quad 0 $$

- W=0, cost(W)=4.67
$$ \frac { { (0\times 1-1) }^{ 2 }\quad +\quad { (0\times 2-2) }^{ 2 }\quad +\quad { (0\times 3-3) }^{ 2 } }{ 3 } \quad =\quad 4.67 $$

- W=2, cost(W)=4.67
$$ \frac { { (2\times 1-1) }^{ 2 }\quad +\quad { (2\times 2-2) }^{ 2 }\quad +\quad { (2\times 3-3) }^{ 2 } }{ 3 } \quad =\quad 4.67 $$

## How to minimize cost?
$$ cost(W)=\frac { 1 }{ m } \sum _{ i=1 }^{ m }{ { (W{x}^{ (i) }-{ y }^{ (i) }) }^{ 2 } } $$ 
![](images/cost_graph2.png?raw=true)

Cost가 가장 작은 W를 찾아내야 한다.

## Gradient descent algorithm
경사를 따라 내려가는 알고리즘

- Minimize cost function
- 많은 Minimiziztion 문제에 사용되고 있다.
- cost(W,b)에 적합합 algorithm
- 강사도가 낮은 방향으로 이동 (경사도를 줄이는 작업...)

$$ W := W - \alpha \frac { 1 }{ m } \sum _{ i=1 }^{ m }{ (W{ x }^{ (i) }-{ y }^{ (i) }) } { x }^{ (i) } $$

## Convex function
Cost function이 convex function인지 확인 필요
![](images/convex_function1.png?raw=true)
![](images/convex_function2.png?raw=true)

### Sample code
Cost graph
~~~
from __future__ import print_function

import tensorflow as tf
from matplotlib import pyplot as plt

# Graph Input
X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_smaples = len(X)

# model weight
W = tf.placeholder(tf.float32)

# Construct a linear model
hypothesis = tf.multiply(X, W)

# Cost function
cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / m

init = tf.global_variables_initializer()

# for graphs
W_val = []
cost_val = []

# Launch the graphs
sess = tf.Session()
sess.run(init)

for i in range(-30, 50):
    print(i * -0.1, sess.run(cost, feed_dict={W: i * 0.1}))
    W_val.append(i * 0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i * 0.1}))

plt.plot(W_val, cost_val, 'ro')
plt.ylabel('cost')
plt.xlabel('W')
plt.show()
~~~

~~~
from __future__ import print_function

import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#######################################################
# hypothesis = W * X + b
hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

#######################################################
# a = tf.Variable(0.1) # Learning rate, alpha
# optimizer = tf.train.GradientDescentOptimizer(a)
# train = optimizer.minimize(cost)

lr = 0.1
descent = W - tf.multiply(lr, tf.reduce_mean(tf.multiply((tf.multiply(W, X) - Y), X)))
train = W.assign(descent)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))
~~~
