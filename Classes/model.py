import random
import numpy as np

class MClassifier:
    """
    Info:
        Rede neural baseada no modelo perceptron de multicamadas (Multi-Layer Perceptron - MLP).
        Por enquanto, a rede neural possui apenas uma camada de entrada, uma oculta e uma de saida.

    Functions: 
        fit: Função para treinar a rede neural.
        predict: Função para predizer o resultado da rede neural.
        initialize_weights: Função para inicializar os pesos da rede neural.
    
    """

    def __init__(self):
        #Inicializa listas para armazenar os pesos.
        self._weights_layer1 = []
        self._weights_layer2 = []
        #Inicializa uma variável para armazenar o erro.
        self._loss = 0

    def fit(self):
        pass

    def predict(self):
        pass

    def initialize_weights(self, x, neurons):
        """
        Info:
            Esta função inicializa os pesos das camadas da rede neural.

        Params:
            x: Features de entrada da rede.
            neurons: Quantidade de neurônios definidos para a camada oculta.
        """
        #Percorre o numero de caracteristicas
        for i in range(len(x[0])):

            #Gera uma lista de valores aleatorios
            generated_weights = [round(random.random(), 3) for j in range(neurons)]

            #Junta os valores aleatorios para formarem os pesos da camada 0
            self._weights_layer1.append(generated_weights)

        #Percorre o numero de neuronios
        for i in range(neurons):

            #Gera uma lista de valores aleatorios
            generated_weights = [round(random.random(), 3)]

            #Junta os valores aleatorios para formarem os pesos da camada 1
            self._weights_layer2.append(generated_weights)

        #Converte a lista em array numpy
        self._weights_layer1 = np.asarray(self._weights_layer1)
        self._weights_layer2 = np.asarray(self._weights_layer2)