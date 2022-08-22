from model import MClassifier
from rich import print
import numpy as np
from model_tools import save, load

#Teste simples com o operador XOR
x_train = np.array([[0, 0], 
                    [0, 1], 
                    [1, 0], 
                    [1, 1]])

y = np.array([[0], [1], [1], [0]])

x_test = np.array([[0, 1], 
                   [1, 0]])

#Instanciando a classificador (Modelo já foi treinado e saldo)
#m = MClassifier()

#Treinando o modelo (Já foi treinado)
#r = m.fit(x_train, y, epoch=5000, neurons=1000, learning_rate=0.1, error_threshold=0.05, moment=1)

#Salvando o modelo (Já foi salvo)
#save(m, "model.pkl")

#Carregando o modelo
loaded_model = load("model.pkl")

#Testando o modelo
test = loaded_model.predict(x_test)
print("-----------------Teste de predição-----------------")
print("Predição do teste [[0, 0], [0, 1]]\nRetorno: {}".format(test))
print("---------------------------------------------------")

#Metrics
print("---------------------Metrics-----------------------")
print("Accuraccy: {}".format(loaded_model._accuracy))
print("Precision: {}".format(loaded_model._precision))
print("Recall: {}".format(loaded_model._recall))
print("F1: {}".format(loaded_model._f1))
print("Kappa: {}".format(loaded_model._kappa))
print("---------------------------------------------------")




