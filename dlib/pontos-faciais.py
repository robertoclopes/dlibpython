import dlib
import cv2
import numpy as np

def imprimePontos(imagem, pontosFaciais):   #percorre os 68 pontos faciais (x, y)
    for p in pontosFaciais.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0, 255, 0), 2)   #metodo para desenhar circulo de raio 2

def imprimeNumeros(imagem, pontosFaciais):
    for i, p in enumerate(pontosFaciais.parts()):
        cv2.putText(imagem, str(i), (p.x, p.y), fonte, .55, (0, 0, 255), 1)

def imprimeLinhas(imagem, pontosFaciais):
    p68 = [[0, 16, False], # linha do queixo
           [17, 21, False], # sombrancelha direita
           [22, 26, False], # sombancelha esquerda
           [27, 30, False], # ponte nasal
           [30, 35, True], # nariz inferior
           [36, 41, True], # olho esquerdo
           [42, 47, True], # olho direito
           [48, 59, True], # labio externo
           [60, 67, True]] # labio interno
    for k in range(0, len(p68)):
        pontos = []
        for i in range(p68[k][0], p68[k][1] + 1):
            ponto = [pontosFaciais.part(i).x, pontosFaciais.part(i).y]
            pontos.append(ponto)    #adiciona ponto por ponto capturado em p68
        pontos = np.array(pontos, dtype=np.int32)   #converte em um array numpy (lib para operacoes matematicas) de numeros inteiros
        cv2.polylines(imagem, [pontos], p68[k][2], (255, 0, 0), 2)  #desenha linhas por cada faixa de pontos
                                                                    #verifica ponto se True desenha linha fechada

fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL
imagem = cv2.imread("fotos/treinamento/ronald.0.1.jpg")
#imagem = cv2.imread("fotos/grupo.0.jpg")
#imagem = cv2.imread("fotos/grupo.1.jpg")
#imagem = cv2.imread("fotos/grupo.2.jpg")
#imagem = cv2.imread("fotos/grupo.3.jpg")
#imagem = cv2.imread("fotos/grupo.4.jpg")
#imagem = cv2.imread("fotos/grupo.5.jpg")
#imagem = cv2.imread("fotos/grupo.6.jpg")
#imagem = cv2.imread("fotos/grupo.7.jpg")

detectorFace = dlib.get_frontal_face_detector() #detecta a face
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat") #detecta os pontos faciais na regiao de interesse ROI
                                                                                        #arquivo já treinado com o preditor com os 68pontos de face
facesDetectadas = detectorFace(imagem, 2)   #bounding boxes
for face in facesDetectadas:
    pontos = detectorPontos(imagem, face)   #aplica o detector na bounding box (regiao de interesse)
    print(pontos.parts())   #pontos (x, y)
    print(len(pontos.parts()))  #68 pontos
    #imprimePontos(imagem, pontos)
    #imprimeNumeros(imagem, pontos)
    imprimeLinhas(imagem, pontos)

cv2.imshow("Pontos faciais", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()