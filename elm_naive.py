#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 16:29:09 2014

@author: akusok
"""

import numpy as np
import os
from hpelm import ELM
import hpelm

curdir = os.path.dirname(__file__)
pX = os.path.join(curdir, "../dataset_tests/iris/iris_data.txt")
pY = os.path.join(curdir, "../dataset_tests/iris/iris_classes.txt")

X = np.loadtxt(pX)
Y = np.loadtxt(pY)



elm = ELM(4,3)
elm.add_neurons(15, "sigm")
elm.train(X, Y, "c")
Yh = elm.predict(X)
print(len(Yh))
print Yh.argmax(1)
acc = float(np.sum(Y.argmax(1) == Yh.argmax(1))) / Y.shape[0]
print(acc)
#print "Iris dataset training error: %.1f%%" % (100-acc*100)


'''

#test_data.py test_SigmClassification_Iris_BetterThanNaive
elm = hpelm.ELM(4, 3)
elm.add_neurons(10, "sigm")
elm.train(X, Y, 'c')
Y1 = elm.predict(X)
print(Y1)
err = elm.error(Y1, Y)
print "-----------------------------------------"
print err
print "-----------------------------------------"

#test_data.py test_RBFClassification_Iris_BetterThanNaive

elm = hpelm.ELM(4, 3)
elm.add_neurons(10, "rbf_l2")
elm.train(X, Y, 'c')
Y1 = elm.predict(X)
err = elm.error(Y1, Y)
print "-----------------------------------------"
print err
print "-----------------------------------------"




'''



