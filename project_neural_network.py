# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 11:34:32 2021

@author: rashi
"""
import sys 

from pyspark import SparkContext
sc = SparkContext.getOrCreate()

data = sc.textFile(sys.argv[1])
header = data.first() 
data = data.filter(lambda row: row != header)

data_rd = data.map(lambda x: list(map(int, x.split(","))))

import numpy as np

def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_der(x):
  s = sigmoid(x)
  return s*(1-s)

def cost_sse(y_pred, y):
  return 0.5 * np.sum(np.power(y_pred - y, 2))

def forward(x, w, b):
  return np.dot(x, w) + b

def predict(x, w1, b1, w2, b2):
  return sigmoid(forward(sigmoid(forward(x, w1, b1)), w2, b2))


def y_trans(y):
  y2 = np.zeros(10)
  y2[y] = 1
  return y2

lr = 0.1
input_layer = 784
hidden_layer = 200
output_layer = 10

w1 = np.random.rand(input_layer, hidden_layer) - 0.5 
w2 = np.random.rand(hidden_layer, output_layer) - 0.5 
b1 = np.random.rand(1, hidden_layer) - 0.5 
b2 = np.random.rand(1, output_layer) - 0.5
cost_list = []
acc_list = []
w1_list=[]
w2_list = []
b1_list = []
b2_list = []



for i in range(450):
  sample = data_rd.sample(False, 0.05)
  grad2 = sample.map(lambda x: (x[0], forward(x[1:], w1, b1), x[1:]))\
        .map(lambda x: (x[0], x[1], sigmoid(x[1]), x[2]))\
        .map(lambda x: (x[0], x[1], x[2], forward(x[2], w2, b2), x[3]))\
        .map(lambda x: (x[0], x[1], x[2], sigmoid(x[3]), x[4]))\
        .map(lambda x: (x[0], x[1], x[2], x[3], x[3] - y_trans(x[0]), x[4]))\
        .map(lambda x: (x[0], x[1], x[2], x[3], x[4], np.dot(x[4], w2.T)*x[2]*(1-x[2]), x[5]))\
        .map(lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], np.dot(np.array([x[6]]).T, x[5]), x[6]))\
        .map(lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6], np.dot(np.array(x[2]).T, x[4]), x[7]))\
        .map(lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], np.dot(x[4], w2.T)*sigmoid_der(x[1]), x[8]))\
        .map(lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], (x[3] - y_trans(x[0])*x[3]*(1-x[3])), 
                        x[9],1, int(x[0] == np.argmax(x[3])), cost_sse(np.argmax(x[3]), x[0])))\
        .reduce(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3], x[4] 
                             + y[4], x[5] + y[5], x[6] + y[6], x[7] + y[7], x[8]+y[8], x[9]+y[9], x[10]+y[10], x[11]+y[11], x[12]+y[12], x[13]+y[13]))
  n = grad2[-3]
  cost = grad2[-1]/n
  print(cost)
  acc = grad2[-2]/n
  cost_list.append(cost)
  acc_list.append(acc)
  dw1 = grad2[6]/n
  dw2 = grad2[7]/n
  db1 = grad2[8]/n
  db2 = grad2[9]/n 

  w1_list.append(w1)
  w2_list.append(w2)
  b1_list.append(b1)
  b2_list.append(b2)

  w1 -= lr*dw1
  w2 -= lr*dw2
  b1 -= lr*db1
  b2 -= lr*db2
  
  
  
test_data = sc.textFile(sys.argv[2])
header = test_data.first() 
test = test_data.filter(lambda row: row != header)

test_rd = test.map(lambda x: list(map(int, x.split(","))))

test_pred = test_rd.map(lambda x: (x[0], predict(x[1:], w1_list[-1], b1_list[-1], w2_list[-1], b2_list[-1])))\
            .map(lambda x: (1, int(x[0] == np.argmax(x[1]))))\
            .reduce(lambda x, y: (x[0] + y[0], x[1]+y[1]))
            
print("accuracy: ")
print(test_pred[1]/test_pred[0])

max_ind = np.argmax(acc_list)


test_pred1 = test_rd.map(lambda x: (x[0], predict(x[1:], w1_list[max_ind], b1_list[max_ind], w2_list[max_ind], b2_list[max_ind])))\
            .map(lambda x: (1, int(x[0] == np.argmax(x[1]))))\
            .reduce(lambda x, y: (x[0] + y[0], x[1]+y[1]))
            
accuracy = test_pred1[1]/test_pred1[0]

print("Accuracy: ")
print(accuracy)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(list(range(450)), cost_list)
plt.title("Cost per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(list(range(450)), acc_list)
plt.title("Accuracy per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

