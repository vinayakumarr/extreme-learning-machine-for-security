#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 16:29:09 2014

@author: vinayakumar R
"""

import numpy as np
import os
from hpelm import ELM
import hpelm

curdir = os.path.dirname(__file__)
pX = os.path.join(curdir, "../dataset_tests/iris/maldata1.txt")
pY = os.path.join(curdir, "../dataset_tests/iris/mallabel.txt")

#X = np.genfromtxt(pX, dtype= None,delimiter=" ")
X = np.loadtxt(pX)
Y = np.loadtxt(pY)
print(type(X))

print "sigmoid with multi class error"
elm = ELM(1804,10)
elm.add_neurons(150, "sigm")
elm.train(X, Y, "c")
Yh = elm.predict(X)
print Yh
acc = float(np.sum(Y.argmax(1) == Yh.argmax(1))) / Y.shape[0]
print "malware dataset training error: %.1f%%" % (100-acc*100)

print "sigmoid with MSE"
elm = hpelm.ELM(1804, 10)
elm.add_neurons(150, "sigm")
elm.train(X, Y)
Y1 = elm.predict(X)
err = elm.error(Y1, Y)
print err

print "rbf_12 with multi class error"
elm = hpelm.ELM(1804, 10)
elm.add_neurons(150, "rbf_l2")
elm.train(X, Y, 'c')
Y1 = elm.predict(X)
err = elm.error(Y1, Y)
print err


print "rbf_11 with multi class error"
elm = hpelm.ELM(1804, 10)
elm.add_neurons(150, "rbf_l1")
elm.train(X, Y, 'c')
Y1 = elm.predict(X)
err = elm.error(Y1, Y)
print err
print(str(elm))

print "rbf_linf with multi class error"
elm = hpelm.ELM(1804, 10)
elm.add_neurons(150, "rbf_linf")
elm.train(X, Y, 'c')
Y1 = elm.predict(X)
err = elm.error(Y1, Y)
print err


print "tanh with multi class error"
elm = hpelm.ELM(1804, 10)
elm.add_neurons(150, "tanh")
elm.train(X, Y, 'c')
Y1 = elm.predict(X)
err = elm.error(Y1, Y)
print err

print "lin with multi class error"
elm = hpelm.ELM(1804, 10)
elm.add_neurons(150, "lin")
elm.train(X, Y, 'c')
Y1 = elm.predict(X)
err = elm.error(Y1, Y)
print err






