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
from sklearn import metrics
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
import scipy as sp



def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


curdir = os.path.dirname(__file__)
pX = os.path.join(curdir, "train/NORM_train_mal.txt")
pY = os.path.join(curdir, "train/classlabelonehot.txt")
pX1 = os.path.join(curdir, "test/NORM_test_mal.txt")
pY1 = os.path.join(curdir, "train/classlabel.txt")


X = np.loadtxt(pX)
Y = np.loadtxt(pY)
X1 = np.loadtxt(pX1)
Y1 = np.loadtxt(pY1)


result = open("data/result/sigmulti.txt", "w")

result.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc")
result.write("\n")



result1 = open("data/result/sigmse.txt", "w")

result1.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc")
result1.write("\n")



result2 = open("data/result/rbf12.txt", "w")

result2.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc")
result2.write("\n")


result2a = open("data/result/rbf12mse.txt", "w")
result2a.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc")
result2a.write("\n")


result3 = open("data/result/rbf11.txt", "w")

result3.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc")
result3.write("\n")


result3a = open("data/result/rbf11mse.txt", "w")
result3a.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc")
result3a.write("\n")



result4 = open("data/result/rbflinf.txt", "w")

result4.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc")
result4.write("\n")


result4a = open("data/result/rbflinfmse.txt", "w")
result4a.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc")
result4a.write("\n")



result5 = open("data/result/tanh.txt", "w")

result5.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc")
result5.write("\n")


result5a = open("data/result/tanhmse.txt", "w")
result5a.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc")
result5a.write("\n")


result6 = open("data/result/lin.txt", "w")

result6.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc")
result6.write("\n")




result6a = open("data/result/linmse.txt", "w")
result6a.write("accuracy" +  "\t" +  "precision" + "\t" + "recall" + "\t" + "f1-measure" +"\t" + " mse" + "\t" + " mae" + "\t" + "auc")
result6a.write("\n")

thefile = open('data/result/sigmultilogloss.txt', 'w')
thefile1 = open('data/result/sigmselogloss.txt', 'w')
thefile2 = open('data/result/rbf12multilogloss.txt', 'w')
thefile3 = open('data/result/rbf12mselogloss.txt', 'w')
thefile4 = open('data/result/rbf11multilogloss.txt', 'w')
thefile5 = open('data/result/rbf11mselogloss.txt', 'w')
thefile6 = open('data/result/rbflinfmultilogloss.txt', 'w')
thefile7 = open('data/result/rbflinfmselogloss.txt', 'w')
thefile8 = open('data/result/tanhmultilogloss.txt', 'w')
thefile9 = open('data/result/tanhmselogloss.txt', 'w')
thefile10 = open('data/result/linmultilogloss.txt', 'w')
thefile11 = open('data/result/linmselogloss.txt', 'w')


print "sigmoid with multi class error"
elm = ELM(1804,9)
elm.add_neurons(900, "sigm")
elm.train(X, Y, "c","CV", k=5)
r1 = elm.predict(X)
print("r1 shape")
print(r1[0])
print(str(elm))
print("performance measures")
result.write(str(elm))
result.write("\n")

print("Logloss")
logloss1 = logloss(Y,r1)
print(type(logloss1))
print(logloss1)
print("data logloss")

for item in logloss1:
  thefile.write("%s\n" % logloss1)

np.savetxt('data/result/sigmulpredicted.txt', r1)
r1=r1.argmax(1)
print(r1)
accuracy = accuracy_score(Y1, r1)
print(accuracy)
recall = recall_score(Y1, r1, average="weighted")
precision = precision_score(Y1, r1 , average="weighted")
f1 = f1_score(Y1, r1 , average="weighted")
mse = mean_squared_error(Y1, r1)
mae = mean_absolute_error(Y1, r1)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r1,pos_label=2)
auc = metrics.auc(fpr, tpr)
print(metrics.classification_report(Y1, r1))
print(metrics.confusion_matrix(Y1, r1))
print("\n")

result.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc)
result.write("\n")

np.savetxt('data/result/sigmultiother.txt', (fpr,tpr,thresholds), fmt='%10.3f')




print "sigmoid with MSE"
elm = hpelm.ELM(1804,9)
elm.add_neurons(900, "sigm")
elm.train(X, Y,"CV", k=5)
r2 = elm.predict(X)
print(str(elm))


result1.write(str(elm))
result1.write("\n")

logloss2 = logloss(Y,r2)
for item in logloss2:
  thefile1.write("%s\n" % logloss2)

np.savetxt('data/result/sigmsepredicted.txt', r2)


r2=r2.argmax(1)
accuracy = accuracy_score(Y1, r2)
print(accuracy)
recall = recall_score(Y1, r2, average="weighted")
precision = precision_score(Y1, r2 , average="weighted")
f1 = f1_score(Y1, r2 , average="weighted")
mse = mean_squared_error(Y1, r2)
mae = mean_absolute_error(Y1, r2)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r2,pos_label=2)
auc = metrics.auc(fpr, tpr)
print(metrics.classification_report(Y1, r2))
print(metrics.confusion_matrix(Y1, r2))
print("\n")

result1.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc)
result1.write("\n")

np.savetxt('data/result/sigmseother.txt', (fpr,tpr,thresholds), fmt='%10.3f')
print("\n")

print "rbf_12 with multi class error"
elm = hpelm.ELM(1804,9)
elm.add_neurons(900, "rbf_l2")
elm.train(X, Y, 'c',"CV", k=10)
r3 = elm.predict(X)
print(str(elm))

result2.write(str(elm))
result2.write("\n")

logloss3 = logloss(Y,r3)
for item in logloss3:
  thefile2.write("%s\n" % logloss3)

np.savetxt('data/result/rbf12mulpredicted.txt', r3)


r3=r3.argmax(1)
accuracy = accuracy_score(Y1, r3)
print(accuracy)
recall = recall_score(Y1, r3, average="weighted")
precision = precision_score(Y1, r3 , average="weighted")
f1 = f1_score(Y1, r3 , average="weighted")
mse = mean_squared_error(Y1, r3)
mae = mean_absolute_error(Y1, r3)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r3,pos_label=2)
auc = metrics.auc(fpr, tpr)
print(metrics.classification_report(Y1, r3))
print(metrics.confusion_matrix(Y1, r3))
print("\n")

result2.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc)
result2.write("\n")

np.savetxt('data/result/rbf12other.txt', (fpr,tpr,thresholds), fmt='%10.3f')

print("\n")





print "rbf_12 with MSE"
elm = hpelm.ELM(1804,9)
elm.add_neurons(900, "rbf_l2")
elm.train(X, Y, "CV", k=5)
r3a = elm.predict(X)
print(str(elm))

result2a.write(str(elm))
result2a.write("\n")

logloss4 = logloss(Y,r3a)
for item in logloss4:
  thefile3.write("%s\n" % logloss4)

np.savetxt('data/result/rbf12msepredicted.txt', r3a)


r3a=r3a.argmax(1)
accuracy = accuracy_score(Y1, r3a)
print(accuracy)
recall = recall_score(Y1, r3a, average="weighted")
precision = precision_score(Y1, r3a , average="weighted")
f1 = f1_score(Y1, r3a , average="weighted")
mse = mean_squared_error(Y1, r3a)
mae = mean_absolute_error(Y1, r3a)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r3a,pos_label=2)
auc = metrics.auc(fpr, tpr)
print(metrics.classification_report(Y1, r3a))
print(metrics.confusion_matrix(Y1, r3a))
print("\n")

result2a.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc)
result2a.write("\n")

np.savetxt('data/result/rbf12mseother.txt', (fpr,tpr,thresholds), fmt='%10.3f')

print("\n")



print "rbf_11 with multi class error"
elm = hpelm.ELM(1804,9)
elm.add_neurons(900, "rbf_l1")
elm.train(X, Y, 'c',"CV", k=5)
r4 = elm.predict(X)
print(str(elm))

result3.write(str(elm))
result3.write("\n")


logloss5 = logloss(Y,r4)
for item in logloss5:
  thefile4.write("%s\n" % logloss5)

np.savetxt('data/result/rbf11mulpredicted.txt', r4)



r4=r4.argmax(1)
accuracy = accuracy_score(Y1, r4)
print(accuracy)
recall = recall_score(Y1, r4, average="weighted")
precision = precision_score(Y1, r4 , average="weighted")
f1 = f1_score(Y1, r4 , average="weighted")
mse = mean_squared_error(Y1, r4)
mae = mean_absolute_error(Y1, r4)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r4,pos_label=2)
auc = metrics.auc(fpr, tpr)
print(metrics.classification_report(Y1, r4))
print(metrics.confusion_matrix(Y1, r4))
print("\n")

result3.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc)
result3.write("\n")

np.savetxt('data/result/rbf11other.txt', (fpr,tpr,thresholds), fmt='%10.3f')



print("\n")



print "rbf_11 with MSE"
elm = hpelm.ELM(1804,9)
elm.add_neurons(900, "rbf_l1")
elm.train(X, Y, "CV", k=5)
r4a = elm.predict(X)
print(str(elm))

result3a.write(str(elm))
result3a.write("\n")


logloss6 = logloss(Y,r4a)
for item in logloss6:
  thefile5.write("%s\n" % logloss6)

np.savetxt('data/result/rbf11msepredicted.txt', r4a)



r4a=r4a.argmax(1)
accuracy = accuracy_score(Y1, r4a)
print(accuracy)
recall = recall_score(Y1, r4a, average="weighted")
precision = precision_score(Y1, r4a , average="weighted")
f1 = f1_score(Y1, r4a , average="weighted")
mse = mean_squared_error(Y1, r4a)
mae = mean_absolute_error(Y1, r4a)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r4a,pos_label=2)
auc = metrics.auc(fpr, tpr)
print(metrics.classification_report(Y1, r4a))
print(metrics.confusion_matrix(Y1, r4a))
print("\n")

result3a.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc)
result3a.write("\n")

np.savetxt('data/result/rbf11mseother.txt', (fpr,tpr,thresholds), fmt='%10.3f')



print("\n")








print "rbf_linf with multi class error"
elm = hpelm.ELM(1804,9)
elm.add_neurons(900, "rbf_linf")
elm.train(X, Y, 'c',"CV", k=5)
r5 = elm.predict(X)
print(str(elm))

result4.write(str(elm))
result4.write("\n")

logloss7 = logloss(Y,r5)
for item in logloss7:
  thefile6.write("%s\n" % logloss7)

np.savetxt('data/result/rbflinfmulpredicted.txt', r5)


r5=r5.argmax(1)
accuracy = accuracy_score(Y1, r5)
print(accuracy)
recall = recall_score(Y1, r5, average="weighted")
precision = precision_score(Y1, r5 , average="weighted")
f1 = f1_score(Y1, r5 , average="weighted")
mse = mean_squared_error(Y1, r5)
mae = mean_absolute_error(Y1, r5)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r5,pos_label=2)
auc = metrics.auc(fpr, tpr)
print(metrics.classification_report(Y1, r5))
print(metrics.confusion_matrix(Y1, r5))
print("\n")

result4.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc)
result4.write("\n")

np.savetxt('data/result/rbflinfother.txt', (fpr,tpr,thresholds), fmt='%10.3f')



print("\n")





print "rbf_linf with MSE"
elm = hpelm.ELM(1804,9)
elm.add_neurons(900, "rbf_linf")
elm.train(X, Y, "CV", k=5)
r5a = elm.predict(X)
print(str(elm))

result4a.write(str(elm))
result4a.write("\n")

logloss8 = logloss(Y,r5a)
for item in logloss8:
  thefile7.write("%s\n" % logloss8)

np.savetxt('data/result/rbflinfmsepredicted.txt', r5a)
np.savetxt('data/result/sigmulpredicted.txt', r1)



r5a=r5a.argmax(1)
accuracy = accuracy_score(Y1, r5a)
print(accuracy)
recall = recall_score(Y1, r5a, average="weighted")
precision = precision_score(Y1, r5a , average="weighted")
f1 = f1_score(Y1, r5a , average="weighted")
mse = mean_squared_error(Y1, r5a)
mae = mean_absolute_error(Y1, r5a)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r5a,pos_label=2)
auc = metrics.auc(fpr, tpr)
print(metrics.classification_report(Y1, r5a))
print(metrics.confusion_matrix(Y1, r5a))
print("\n")

result4a.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc)
result4a.write("\n")

np.savetxt('data/result/rbflinmsefother.txt', (fpr,tpr,thresholds), fmt='%10.3f')



print("\n")



print "tanh with multi class error"
elm = hpelm.ELM(1804,9)
elm.add_neurons(900, "tanh")
elm.train(X, Y, 'c',"CV", k=5)
r6 = elm.predict(X)
print(str(elm))

result5.write(str(elm))
result5.write("\n")

logloss9 = logloss(Y,r6)
for item in logloss9:
  thefile8.write("%s\n" % logloss9)

np.savetxt('data/result/tanhmulpredicted.txt', r6)



r6=r6.argmax(1)
accuracy = accuracy_score(Y1, r6)
print(accuracy)
recall = recall_score(Y1, r6, average="weighted")
precision = precision_score(Y1, r6 , average="weighted")
f1 = f1_score(Y1, r6 , average="weighted")
mse = mean_squared_error(Y1, r6)
mae = mean_absolute_error(Y1, r6)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r6,pos_label=2)
auc = metrics.auc(fpr, tpr)
print(metrics.classification_report(Y1, r6))
print(metrics.confusion_matrix(Y1, r6))
print("\n")

result5.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc)
result5.write("\n")

np.savetxt('data/result/tanhother.txt', (fpr,tpr,thresholds), fmt='%10.3f')



print("\n")






print "tanh with MSE"
elm = hpelm.ELM(1804,9)
elm.add_neurons(900, "tanh")
elm.train(X, Y, "CV", k=5)
r6a = elm.predict(X)
print(str(elm))

result5a.write(str(elm))
result5a.write("\n")

logloss10 = logloss(Y,r6a)
for item in logloss10:
  thefile9.write("%s\n" % logloss10)

np.savetxt('data/result/tanhmsepredicted.txt', r6a)



r6a=r6a.argmax(1)
accuracy = accuracy_score(Y1, r6a)
print(accuracy)
recall = recall_score(Y1, r6a, average="weighted")
precision = precision_score(Y1, r6a , average="weighted")
f1 = f1_score(Y1, r6a , average="weighted")
mse = mean_squared_error(Y1, r6a)
mae = mean_absolute_error(Y1, r6a)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r6a,pos_label=2)
auc = metrics.auc(fpr, tpr)
print(metrics.classification_report(Y1, r6a))
print(metrics.confusion_matrix(Y1, r6a))
print("\n")

result5a.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc)
result5a.write("\n")

np.savetxt('data/result/tanhmseother.txt', (fpr,tpr,thresholds), fmt='%10.3f')



print("\n")







print "lin with multi class error"
elm = hpelm.ELM(1804,9)
elm.add_neurons(900, "lin")
elm.train(X, Y, 'c',"CV", k=5)
r7 = elm.predict(X)
print(str(elm))
result6.write(str(elm))
result6.write("\n")

logloss11 = logloss(Y,r7)
for item in logloss11:
  thefile10.write("%s\n" % logloss11)

np.savetxt('data/result/linmulpredicted.txt', r7)




r7=r7.argmax(1)
accuracy = accuracy_score(Y1, r7)
print(accuracy)
recall = recall_score(Y1, r7, average="weighted")
precision = precision_score(Y1, r7 , average="weighted")
f1 = f1_score(Y1, r7 , average="weighted")
mse = mean_squared_error(Y1, r7)
mae = mean_absolute_error(Y1, r7)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r7,pos_label=2)
auc = metrics.auc(fpr, tpr)
print(metrics.classification_report(Y1, r7))
print(metrics.confusion_matrix(Y1, r7))
print("\n")

result6.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc)
result6.write("\n")

np.savetxt('data/result/linfother.txt', (fpr,tpr,thresholds), fmt='%10.3f')

print("\n")






print "lin with MSE"
elm = hpelm.ELM(1804,9)
elm.add_neurons(900, "lin")
elm.train(X, Y,"CV", k=5)
r7a = elm.predict(X)
print(str(elm))
result6a.write(str(elm))
result6a.write("\n")

logloss12 = logloss(Y,r7a)
for item in logloss12:
  thefile11.write("%s\n" % logloss12)

np.savetxt('data/result/linmsepredicted.txt', r7a)



r7a=r7a.argmax(1)
accuracy = accuracy_score(Y1, r7a)
print(accuracy)
recall = recall_score(Y1, r7a, average="weighted")
precision = precision_score(Y1, r7a , average="weighted")
f1 = f1_score(Y1, r7a , average="weighted")
mse = mean_squared_error(Y1, r7a)
mae = mean_absolute_error(Y1, r7a)
fpr, tpr, thresholds = metrics.roc_curve(Y1, r7a,pos_label=2)
auc = metrics.auc(fpr, tpr)
print(metrics.classification_report(Y1, r7a))
print(metrics.confusion_matrix(Y1, r7a))
print("\n")

result6a.write("%1.5f" % accuracy + "\t%1.3f" % precision + "\t%1.3f"%recall + "\t%1.3f" % f1 +"\t%1.5f" % mse +"\t%1.5f" % mae +"\t%1.5f" % auc)
result6a.write("\n")

np.savetxt('data/result/linmsefother.txt', (fpr,tpr,thresholds), fmt='%10.3f')

print("\n")







print("--------------------------------------------------------------------------------------------------------------")

'''








model= hpelm.ELM(1804,9)
model.add_neurons(100,"sigm")
model.add_neurons(50,"lin")


model.train(X, Y)
tth =model.predict(X)
print model.error(tth, Y)
r8 =model.predict(X)
r8=r8.argmax(1)
accuracy = accuracy_score(Y1, r8)
print(accuracy)

print("=============================================================================================================================")



model.train (X , Y , "CV", k = 5)
r9 =model.predict(X)
print(r9)
r9=r9.argmax(1)
accuracy = accuracy_score(Y1, r9)
print(accuracy)


model. train(X, Y, "LOO", "OP")
r10 =model.predict(X)
print(r10)
r10=r10.argmax(1)
accuracy = accuracy_score(Y1, r10)
print(accuracy)


'''


