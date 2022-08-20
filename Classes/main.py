from model import MClassifier
from rich import print
import numpy as np

#Teste simples com o operador XOR
x_train = np.array([[0, 0], 
              [0, 1], 
              [1, 0], 
              [1, 1]])

y = np.array([[0], [1], [1], [0]])

x_test = np.array([[0, 0], 
                   [1, 0]])

#Instanciando a classificador
m = MClassifier()

#Treinando o modelo
r = m.fit(x_train, y, epoch=5000, neurons=1000, learning_rate=0.1, error_threshold=0.05, moment=1)

#Metrics
print("Epochs: {}".format(r))
print("Accuraccy: {}".format(m._accuracy))
print("Precision: {}".format(m._precision))
print("Recall: {}".format(m._recall))
print("F1: {}".format(m._f1))
print("Kappa: {}".format(m._kappa))

#Testando o modelo
test = m.predict(x_test)

print("Predição do teste [0, 0]: {}".format(test))




