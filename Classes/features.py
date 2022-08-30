from skimage.feature import greycomatrix, greycoprops
import numpy as np

class Features:

    @staticmethod
    def GLCM(dataset, steps):
        """
        Info:
            Calcula a matriz de Co-ocorrencia de níveis de cinza de um dataset.
        Params:
            dataset (array numpy, array, imread_collection): Dataset de imagens.
            steps (array): Lista de deslocamentos de distância de pares de pixels.
        Return:
            Matriz isotrópica.
        """
        matrix = []

        for img in dataset:
            matrix0 = greycomatrix(img, [steps], [0], normed=True)
            matrix1 = greycomatrix(img, [steps], [np.pi/4], normed=True)
            matrix2 = greycomatrix(img, [steps], [np.pi/2], normed=True)
            matrix3 = greycomatrix(img, [steps], [3*np.pi/4], normed=True)
            matrix.append((matrix0+matrix1+matrix2+matrix3)/4)
        
        return np.asarray(matrix)
    
    @staticmethod
    def GLCM_props(matrix):
        """
        Info:
            Calcula as propriedades da matriz de Co-ocorrencia de níveis de cinza.
        Params:
            matrix (array): Array de matrizes de Co-ocorrencia de níveis de cinza.
        Return:
            Matriz com as propriedades de contraste, dissimilaridade, homogeneidade, 
            energia, correlação e ASM.
        """
        props = []

        for mat in matrix:
            prop = np.zeros((6))
            prop[0] = greycoprops(mat,'contrast')
            prop[1] = greycoprops(mat,'dissimilarity')
            prop[2] = greycoprops(mat,'homogeneity')
            prop[3] = greycoprops(mat,'energy')
            prop[4] = greycoprops(mat,'correlation')
            prop[5] = greycoprops(mat,'ASM')
            props.append(prop)
        
        return np.asarray(props)
