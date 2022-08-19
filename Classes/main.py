from model import MClassifier
from rich import print
import numpy as np

#Teste simples com o operador XOR
x = np.array([[0, 0], 
              [0, 1], 
              [1, 0], 
              [1, 1]])

y = np.array([[0], [1], [1], [0]])

#Instanciando a classificador
m = MClassifier()
#Treinando o modelo
r = m.fit(x, y, epoch=5000, neurons=1000, learning_rate=0.1, error_threshold=0.05, moment=1)

print("Epochs: {}".format(r))
print("Pesos Layer 1: {}".format(m._weights_layer1))
print("Pesos Layer 2: {}".format(m._weights_layer2))



