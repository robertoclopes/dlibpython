'''
essa fase do treinamento gera dois arquivos:
    npy -   arquivo com os descritores de cada imagem
            cada descritor possui um conjunto com 128 caracteristicas obtidas utilizando filtros (kernels) nas CNN

    pickle  -   arquivo que apresenta o indice (id, nome) de cada imagem
'''

import os
import glob #utilizada para percorrer os arquivos de imagens
import _pickle as cPickle   #utilizada para gravacao do arquivo de treinamento
import dlib
import cv2
import numpy as np

#fase de deteccao
detectorFace = dlib.get_frontal_face_detector() #detector de bounding box (ROI)
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat") #detector de pontos faciais dentro da ROI
reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat") #arquivo treinado para fazer reconhecimento facial utilizadno CNN

indice = {} #dicionario para armazenar o nome do arquivo
idx = 0
descritoresFaciais = None

for arquivo in glob.glob(os.path.join("fotos/treinamento", "*.jpg")):   #arquivo recebe cada uma das imagens
    imagem = cv2.imread(arquivo)    #faz leitura da imagem
    facesDetectadas = detectorFace(imagem, 1)   #recebe bounding boxes e aumenta a escala da imagem
    numeroFacesDetectadas = len(facesDetectadas)    #recebe o numero de faces detectadas dentro de cada imagem
    #print(numeroFacesDetectadas)
    if numeroFacesDetectadas > 1:   #só pode haver uma face em cada imagem para o treinamento
        print("Há mais de uma face na imagem {}".format(arquivo))
        exit(0)
    elif numeroFacesDetectadas < 1: #nao encontrou nenhuma face
        print("Nenhuma face encontrada no arquivo {}".format(arquivo))
        exit(0)

    #extracao de caracteristicas
    for face in facesDetectadas:    #percorre o bounding box de cada face encontrada
        pontosFaciais = detectorPontos(imagem, face)    #extrai os pontos faciais
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)   #descritor CNN computa as principais caracteristicas da face utilizando os pontos faciais e convolucoes (kernel)
        #print(format(arquivo))
        #print(len(descritorFacial))    #128
        #print(descritorFacial)

        listaDescritorFacial = [df for df in descritorFacial]   #converte o descritor de face do formato Dlib para uma lista de tamanho 128
        #print(listaDescritorFacial)

        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64) #converte a lista para um vetor numpy
        #print(npArrayDescritorFacial)

        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]  #aumenta a dimensao do array para 2D criando nova coluna com todos os dados do descritor facial
        #print(npArrayDescritorFacial)

        #matriz de descritores por cada imagem percorida (8, 128)
        if descritoresFaciais is None:
            descritoresFaciais = npArrayDescritorFacial
        else:
            descritoresFaciais = np.concatenate((descritoresFaciais, npArrayDescritorFacial), axis=0)

        indice[idx] = arquivo   #armazena no dicionario os nomes das imagens começando pelo indice "0"
        idx += 1

    #cv2.imshow("Treinamento", imagem)
    #cv2.waitKey(0)

#print("Tamanho: {} Formato: {}".format(len(descritoresFaciais), descritoresFaciais.shape))
#print(descritoresFaciais)
#print(indice)
np.save("recursos/descritores_rn.npy", descritoresFaciais)  #grava os descritores de cada imagem
with open("recursos/indices_rn.pickle", 'wb') as f: #gravando os indices
    cPickle.dump(indice, f) #grava indice no arquivo f

#cv2.destroyAllWindows()