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

"""Simple command-line example for Tensorflow using Linear Regression.

Command-line application that Tensorflow using Linear Regression.
"""
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

