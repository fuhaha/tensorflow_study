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
### [용어]=====================
** 텐서 (tensor) : 명사[수학][물리학]
3차원 공간에 있어서 9개의 성분을 가지며, 좌표 변환에 의해 좌표 성분의 곱과 같은 형의 변환을 받는 양. 예를 들면, 물체의 관성 모멘트나 변형은 이것으로 표시됨.