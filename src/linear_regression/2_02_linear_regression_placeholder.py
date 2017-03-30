#!/usr/bin/env python3

# Copyright 2017 fuhaha Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple command-line example for Translate.

Command-line application that translates some text.
"""
from __future__ import print_function

import tensorflow as tf

# traning data
x_data = [1, 2, 3]
y_data = [2, 3, 4]

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
sess.run(init) # 초기

# Fit the line.
for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data}) # 실행시 값을 넣는다.
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b)) # 실행시 값을 넣는다.

# placeholder를 사용하여서 다른 input_data를 이용하여 재활용 가능한다
print(sess.run(hypothesis, feed_dict={X: 1})) # 결과 예측
print(sess.run(hypothesis, feed_dict={X: 2})) # 결과 예측
print(sess.run(hypothesis, feed_dict={X: 3})) # 결과 예측
print(sess.run(hypothesis, feed_dict={X: 4})) # 결과 예측
print(sess.run(hypothesis, feed_dict={X: 5})) # 결과 예측
