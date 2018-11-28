'''
essa fase do treinamento gera dois arquivos:
    npy -   arquivo com os descritores de cada imagem
            cada descritor possui um conjunto com 128 caracteristicas obtidas utilizando filtros (kernels) nas CNN

    pickle  -   arquivo que apresenta o indice (id, nome) de cada imagem
'''

import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat") #aplica CNN para obeter melhores caracteristicas

#variaveis para comparar com o processo de treinamento
indices = np.load("recursos/indices_rn.pickle") #indices/ labels
descritoresFaciais = np.load("recursos/descritores_rn.npy") #base de dados de treinamento com os descritores
limiar = 0.5    #define a precisao da classificacao

for arquivo in glob.glob(os.path.join("fotos", "*.jpg")):   #percorre todas as imagens de treinamento
    imagem = cv2.imread(arquivo)
    facesDetectadas = detectorFace(imagem, 2)   #aumenta a escala pois as imagens de teste sao menores
    for face in facesDetectadas:    #percorre o bounding box das faces
        e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom())) #pontos do retangulo
        pontosFaciais = detectorPontos(imagem, face)    #68 pontos da ROI
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)   #extrai o descritor facial de cada imagem (128 caracteristicas)
        listaDescritorFacial = [fd for fd in descritorFacial]   #converte pra lista
        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64) #converte pra array numpy
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]  #adiciona uma nova coluna (2D) com 128 caracteristicas

        #aplicacao do algoritmo KNN (calculo da distancia euclidiana)
        distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)    #funcao numpy para calulcar a distancia euclidiana
        print("Distâncias: {}".format(distancias))  #imprime a distancia da face de teste para as 8 faces de treinamento
        minimo = np.argmin(distancias)  #armazena o indice que contem a distancia minima
        print(minimo)
        distanciaMinima = distancias[minimo]    #retorna o valor do indice minimo
        print(distanciaMinima)

        if distanciaMinima <= limiar:   #<= 0,5
            nome = os.path.split(indices[minimo])[1].split(".")[0]  #recebe o label da imagem mais proxima do teste
        else:
            nome = ' '

        cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 255), 2) #desenha retangulo em torno da face
        texto = "{} {:.4f}".format(nome, distanciaMinima)   #formata o valor da distancia encontrada
        cv2.putText(imagem, texto, (d, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 255))  #texto com nome e distancia minima

    cv2.imshow("Detector hog", imagem)
    cv2.waitKey(0)

cv2.destroyAllWindows()