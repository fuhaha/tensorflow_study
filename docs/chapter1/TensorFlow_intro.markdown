# TensorFlow 
작성자 : 전성욱
작성일 : 2017/03/28
email : codetree@gmail.com

---

## 1 TensorFlow 
- TensorFlow is an opne source software library for numerical computation using data flow graphs.
  TensorFlow란 data flow graphs 이용하여 수학적 계산처리를 하기위한 opensource software library 이다.
- python 
  C++로 만들어 졌으며 C/C++과 Python을 지원한다.

## 2 What is a Data Flow Graph?

- Nodes in graph repesent mathematical operations
  Graph내의 node란 어떤 수학적인 계산을 하기 위한 operations
- Edges represent the multidimensional data arrays (tensor) communicated between them.
  Edge란 node간의 전달되어지는 다차원 데아터 arrays(tensor)

~~~
   +--------+               +--------+ 
   | (Node) |----(Edge)-----| (Node) |
   +--------+               +--------+
                                 |
                               (Edge)
                                 |
                             ---------- 
                             | (Node) |
                             ----------
  -- Graph --
~~~
![](images/graph_operation.png?raw=true)
![](images/22f25697-e54e-5928-defc-ea20cd077187.png?raw=true)

## 3. Hello World
EX1)
~~~
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow') # hello -> operation
sess = tf.Session()
print sess.run(hello)
~~~

## 4. Everything is operation!
EX2)
~~~
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

c = a + b

sess = tf.Session()
print sess.run(c)
~~~ 

## 5. Basic opration
## 6. Placeholder
Placeholder : 입력 Data값을 실행시에 적용할수 있도록 적용을 뒤로 미루기 위한 사전에 확보한 저장공간
     즉 모델 operation정의시 placeholder로 정의된 내용을 사용하고 실제 값은 나중에 적용함다.
EX3)
~~~
import tensorflow as tf
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a,b)
mul = tf.multiply(a,b)

with tf.Session() as sess:
	print ("Multiplication with valiables : {}".format(sess.run(mul, feed_dict = {a:2, b:3})))
~~~

## 개념 정리
![](images/programming.jpg?raw=true)

![](images/machineLearning.jpg?raw=true)

### [용어] Wikipedia 참조
** 텐서(tensor) : 수학과 물리학에서 서로 약간 다른 의미로 사용되는 개념이다. 수학의 다중선형대수학 및 미분기하학 등의 분야에서 텐서는 간단히 말하면 다중선형함수이다. 텐서장이란 기하학적 공간의 각 점마다 위 의미의 텐서가 하나씩 붙어 있는 것을 가리키는데, 물리학과 공학 등에서는 텐서장을 단순히 '텐서'라 부르는 경우도 많다.

** Linear Regression : 통계학에서, 선형 회귀(線型回歸, 영어: linear regression)는 종속 변수 y와 한 개 이상의 독립 변수 (또는 설명 변수) X와의 선형 상관 관계를 모델링하는 회귀분석 기법이다. 한 개의 설명 변수에 기반한 경우에는 단순 선형 회귀, 둘 이상의 설명 변수에 기반한 경우에는 다중 선형 회귀라고 한다.