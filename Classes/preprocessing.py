import numpy as np

class Preprocessing:

    @staticmethod
    def normalize(dataset):
        """
        Info:
            Normaliza os valores de um dataset.
        Params:
            dataset (array numpy): Dataset com imagens 2D-dimensional.
        Return:
            Dataset normalizado.
        """
        normalized = []

        for img in dataset:
            normalized.append((img - np.min(img)) / (np.max(img) - np.min(img)))

        return np.asarray(normalized)

    @staticmethod
    def label_encoder(names, class_1, class_2):
        """
        Info:
            Codifica os valores de um dataset entre 0 e 1.
        Params:
            names (array numpy): Lista com o nome das classes de cada imagem.
            class_1 (string): Nome da primeira classe.
            class_2 (string): Nome da segunda classe.
        Return:
            Dataset codificado.
        """
        encoded = []

        for name in names:
        
            if name.lower() == class_1.lower():
                encoded.append([1])

            elif name.lower() == class_2.lower():
                encoded.append([0])

        return np.asarray(encoded)
