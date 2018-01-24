#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
for binary classificatio
http://stackoverflow.com/questions/36248969/how-to-acquire-sensitivity-and-specifictytrue-positive-rate-and-true-negative-r

for multi-class classificatio



Created on Sun July 16 16:29:09 2016

@author: vinayakumar R
"""

import numpy as np
import os
from hpelm import ELM
import hpelm
from sklearn import metrics
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.metrics import confusion_matrix

print("--------------------------------------------Loading-------------------------------------------------------")
curdir = os.path.dirname(__file__)
pX = os.path.join(curdir, "data/NORM_train.txt")
pY = os.path.join(curdir, "data/classlabelonehot.txt")
pX1 = os.path.join(curdir, "data/NORM_test.txt")
pY1 = os.path.join(curdir, "data/corrected.txt")

X = np.loadtxt(pX)
Y = np.loadtxt(pY)
X1 = np.loadtxt(pX1)
Y1 = np.loadtxt(pY1)
Y1 = np.array(Y1)
print(Y1.shape)

result = open("data/result/sigmulti.txt", "w")
result.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc"+"\t" + "tpr" +"\t" + "fpr")
result.write("\n")



result1 = open("data/result/sigmse.txt", "w")
result1.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc"+"\t" + "tpr" +"\t" + "fpr")
result1.write("\n")



result2 = open("data/result/rbf12.txt", "w")
result2.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc"+"\t" + "tpr" +"\t" + "fpr")
result2.write("\n")



result2a = open("data/result/rbf12mse.txt", "w")
result2a.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc"+"\t" + "tpr" +"\t" + "fpr")
result2a.write("\n")




result3 = open("data/result/rbf11.txt", "w")
result3.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc"+"\t" + "tpr" +"\t" + "fpr")
result3.write("\n")



result3a = open("data/result/rbf11mse.txt", "w")
result3a.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc"+"\t" + "tpr" +"\t" + "fpr")
result3a.write("\n")




result4 = open("data/result/rbflinf.txt", "w")
result4.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc"+"\t" + "tpr" +"\t" + "fpr")
result4.write("\n")



result4a = open("data/result/rbflinfmse.txt", "w")
result4a.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc"+"\t" + "tpr" +"\t" + "fpr")
result4a.write("\n")


result5 = open("data/result/tanh.txt", "w")
result5.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc"+"\t" + "tpr" +"\t" + "fpr")
result5.write("\n")



result5a = open("data/result/tanhmse.txt", "w")
result5a.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc"+"\t" + "tpr" +"\t" + "fpr")
result5a.write("\n")


result6 = open("data/result/lin.txt", "w")
result6.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc"+"\t" + "tpr" +"\t" + "fpr")
result6.write("\n")



result6a = open("data/result/linmse.txt", "w")
result6a.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc"+"\t" + "pr" +"\t" + "fpr")
result6a.write("\n")




print "sigmoid with multi class error"
elm = ELM(41,41)
elm.add_neurons(40, "sigm")
elm.train(X, Y, "c")
r1 = elm.predict(X1)
result.write(str(elm))
result.write("\n")

r1=r1.argmax(1)
accuracy = accuracy_score(Y1, r1)
recall = recall_score(Y1, r1, average="weighted")
precision = precision_score(Y1, r1 , average="weighted")
f1 = f1_score(Y1, r1 , average="weighted")
mse = mean_squared_error(Y1, r1)
mae = mean_absolute_error(Y1, r1)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r1,pos_label=9)
auc = metrics.auc(fpr, tpr)


result.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc +"\t%1.5f" % tpr +"\t%1.5f" % fpr)
result.write("\n")

np.savetxt('data/result/sigmultiother.txt', (fpr,tpr,thresholds), fmt='%10.3f')


'''
print "sigmoid with MSE"
elm = hpelm.ELM(41, 23)
elm.add_neurons(40, "sigm")
elm.train(X, Y)
r2 = elm.predict(X1)
result1.write(str(elm))
result1.write("\n")

r2=r2.argmax(1)
accuracy = accuracy_score(Y1, r2)
recall = recall_score(Y1, r2, average="weighted")
precision = precision_score(Y1, r2 , average="weighted")
f1 = f1_score(Y1, r2 , average="weighted")
mse = mean_squared_error(Y1, r2)
mae = mean_absolute_error(Y1, r2)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r2,pos_label=9)
auc = metrics.auc(fpr, tpr)


result1.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc +"\t%1.5f" % tpr+"\t%1.5f" % fpr)
result1.write("\n")



np.savetxt('data/result/sigmseother.txt', (fpr,tpr,thresholds), fmt='%10.3f')
v = metrics.classification_report(Y1, r2)





print "rbf_12 with multi class error"
elm = hpelm.ELM(41, 23)
elm.add_neurons(40, "rbf_l2")
elm.train(X, Y, 'c')
r3 = elm.predict(X1)
result2.write(str(elm))
result2.write("\n")

r3=r3.argmax(1)
accuracy = accuracy_score(Y1, r3)
recall = recall_score(Y1, r3, average="weighted")
precision = precision_score(Y1, r3 , average="weighted")
f1 = f1_score(Y1, r3 , average="weighted")
mse = mean_squared_error(Y1, r3)
mae = mean_absolute_error(Y1, r3)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r3,pos_label=9)
auc = metrics.auc(fpr, tpr)


result2.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc+"\t%1.5f" % tpr+"\t%1.5f" % fpr)
result2.write("\n")

np.savetxt('data/result/rbf12multiother.txt', (fpr,tpr,thresholds), fmt='%10.3f')




print "rbf_12 with MSE"
elm = hpelm.ELM(41, 23)
elm.add_neurons(40, "rbf_l2")
elm.train(X, Y)
r3a = elm.predict(X1)
result2a.write(str(elm))
result2a.write("\n")

r3a=r3a.argmax(1)
accuracy = accuracy_score(Y1, r3a)
recall = recall_score(Y1, r3a, average="weighted")
precision = precision_score(Y1, r3a , average="weighted")
f1 = f1_score(Y1, r3a , average="weighted")
mse = mean_squared_error(Y1, r3a)
mae = mean_absolute_error(Y1, r3a)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r3a,pos_label=9)
auc = metrics.auc(fpr, tpr)


result2a.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc+"\t%1.5f" % tpr+"\t%1.5f" % fpr)
result2a.write("\n")

np.savetxt('data/result/rbf12mseother.txt', (fpr,tpr,thresholds), fmt='%10.3f')







print "rbf_11 with multi class error"
elm = hpelm.ELM(41, 23)
elm.add_neurons(40, "rbf_l1")
elm.train(X, Y, 'c')
r4 = elm.predict(X1)
result3.write(str(elm))
result3.write("\n")

r4=r4.argmax(1)
accuracy = accuracy_score(Y1, r4)
recall = recall_score(Y1, r4, average="weighted")
precision = precision_score(Y1, r4 , average="weighted")
f1 = f1_score(Y1, r4 , average="weighted")
mse = mean_squared_error(Y1, r4)
mae = mean_absolute_error(Y1, r4)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r4,pos_label=9)
auc = metrics.auc(fpr, tpr)

result3.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc+"\t%1.5f" % tpr+"\t%1.5f" % fpr)
result3.write("\n")

np.savetxt('data/result/rbf11multiother.txt', (fpr,tpr,thresholds), fmt='%10.3f')




print "rbf_11 with MSE"
elm = hpelm.ELM(41, 23)
elm.add_neurons(40, "rbf_l1")
elm.train(X, Y)
r4a = elm.predict(X1)
result3a.write(str(elm))
result3a.write("\n")

r4a=r4a.argmax(1)
accuracy = accuracy_score(Y1, r4a)
recall = recall_score(Y1, r4a, average="weighted")
precision = precision_score(Y1, r4a , average="weighted")
f1a = f1_score(Y1, r4a , average="weighted")
mse = mean_squared_error(Y1, r4a)
mae = mean_absolute_error(Y1, r4a)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r4a,pos_label=9)
auc = metrics.auc(fpr, tpr)


result3a.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc+"\t%1.5f" % tpr+"\t%1.5f" % fpr)
result3a.write("\n")

np.savetxt('data/result/rbf11mseother.txt', (fpr,tpr,thresholds), fmt='%10.3f')





print "rbf_linf with Multi class error"
elm = hpelm.ELM(41, 23)
elm.add_neurons(40, "rbf_linf")
elm.train(X, Y, 'c')
r5 = elm.predict(X1)
result4.write(str(elm))
result4.write("\n")

r5=r5.argmax(1)
accuracy = accuracy_score(Y1, r5)
recall = recall_score(Y1, r5, average="weighted")
precision = precision_score(Y1, r5 , average="weighted")
f1 = f1_score(Y1, r5 , average="weighted")
mse = mean_squared_error(Y1, r5)
mae = mean_absolute_error(Y1, r5)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r5,pos_label=9)
auc = metrics.auc(fpr, tpr)


result4.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc+"\t%1.5f" % tpr+"\t%1.5f" % fpr)
result4.write("\n")

np.savetxt('data/result/rbflinfmultiother.txt', (fpr,tpr,thresholds), fmt='%10.3f')




print "rbf_linf with MSE"
elm = hpelm.ELM(41, 23)
elm.add_neurons(40, "rbf_linf")
elm.train(X, Y)
r5a = elm.predict(X1)
result4a.write(str(elm))
result4a.write("\n")

r5a=r5a.argmax(1)
accuracy = accuracy_score(Y1, r5a)
recall = recall_score(Y1, r5a, average="weighted")
precision = precision_score(Y1, r5a , average="weighted")
f1 = f1_score(Y1, r5a , average="weighted")
mse = mean_squared_error(Y1, r5a)
mae = mean_absolute_error(Y1, r5a)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r5a,pos_label=9)
auc = metrics.auc(fpr, tpr)


result4a.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc+"\t%1.5f" % tpr+"\t%1.5f" % fpr)
result4a.write("\n")

np.savetxt('data/result/rbflinmsefother.txt', (fpr,tpr,thresholds), fmt='%10.3f')



print "tanh with multi class error"
elm = hpelm.ELM(41, 23)
elm.add_neurons(40, "tanh")
elm.train(X, Y, 'c')
r6 = elm.predict(X1)
result5.write(str(elm))
result5.write("\n")

r6=r6.argmax(1)
accuracy = accuracy_score(Y1, r6)
recall = recall_score(Y1, r6, average="weighted")
precision = precision_score(Y1, r6 , average="weighted")
f1 = f1_score(Y1, r6 , average="weighted")
mse = mean_squared_error(Y1, r6)
mae = mean_absolute_error(Y1, r6)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r6,pos_label=9)
auc = metrics.auc(fpr, tpr)


result5.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc+"\t%1.5f" % tpr+"\t%1.5f" % fpr)
result5.write("\n")

np.savetxt('data/result/tanhmultiother.txt', (fpr,tpr,thresholds), fmt='%10.3f')




print "tanh with MSE"
elm = hpelm.ELM(41, 23)
elm.add_neurons(40, "tanh")
elm.train(X, Y)
r6a = elm.predict(X1)
result5a.write(str(elm))
result5a.write("\n")

r6a=r6a.argmax(1)
accuracy = accuracy_score(Y1, r6a)
recall = recall_score(Y1, r6a, average="weighted")
precision = precision_score(Y1, r6a , average="weighted")
f1 = f1_score(Y1, r6a , average="weighted")
mse = mean_squared_error(Y1, r6a)
mae = mean_absolute_error(Y1, r6a)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r6a,pos_label=9)
auc = metrics.auc(fpr, tpr)



result5a.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc+"\t%1.5f" % tpr+"\t%1.5f" % fpr)
result5a.write("\n")

np.savetxt('data/result/tanhmseother.txt', (fpr,tpr,thresholds), fmt='%10.3f')





print "lin with multi class error"
elm = hpelm.ELM(41, 23)
elm.add_neurons(40, "lin")
elm.train(X, Y, 'c')
r7 = elm.predict(X1)
result6.write(str(elm))
result6.write("\n")

r7=r7.argmax(1)
accuracy = accuracy_score(Y1, r7)
recall = recall_score(Y1, r7, average="weighted")
precision = precision_score(Y1, r7 , average="weighted")
f1 = f1_score(Y1, r7 , average="weighted")
mse = mean_squared_error(Y1, r7)
mae = mean_absolute_error(Y1, r7)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r7,pos_label=9)
auc = metrics.auc(fpr, tpr)


result6.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc+"\t%1.5f" % tpr+"\t%1.5f" % fpr)
result6.write("\n")

np.savetxt('data/result/linmultifother.txt', (fpr,tpr,thresholds), fmt='%10.3f')





print "lin with MSE"
elm = hpelm.ELM(41, 23)
elm.add_neurons(40, "lin")
elm.train(X, Y)
r7a = elm.predict(X1)
result6a.write(str(elm))
result6a.write("\n")

r7a=r7a.argmax(1)
accuracy = accuracy_score(Y1, r7a)
recall = recall_score(Y1, r7a, average="weighted")
precision = precision_score(Y1, r7a , average="weighted")
f1 = f1_score(Y1, r7a , average="weighted")
mse = mean_squared_error(Y1, r7a)
mae = mean_absolute_error(Y1, r7a)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r7a,pos_label=9)
auc = metrics.auc(fpr, tpr)


result6a.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc+"\t%1.5f" % tpr+"\t%1.5f" % fpr)
result6a.write("\n")

np.savetxt('data/result/linmsefother.txt', (fpr,tpr,thresholds), fmt='%10.3f')

'''




