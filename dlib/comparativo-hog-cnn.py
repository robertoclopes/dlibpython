import cv2
import dlib

imagem = cv2.imread("fotos/grupo.0.jpg")    #HOG melhor
#imagem = cv2.imread("fotos/grupo.1.jpg")   #CNN detectou mais
#imagem = cv2.imread("fotos/grupo.2.jpg")   #HOG detectou mais
#imagem = cv2.imread("fotos/grupo.3.jpg")   #HOG com maior valor de confianca
#imagem = cv2.imread("fotos/grupo.4.jpg")   #CNN ligeiramente melhor
#imagem = cv2.imread("fotos/grupo.5.jpg")   #HOG detectou mais
#imagem = cv2.imread("fotos/grupo.6.jpg")   #HOG detectou mais
#imagem = cv2.imread("fotos/grupo.7.jpg")   #HOG ligeiramente melhor

detectorHog = dlib.get_frontal_face_detector()
facesDetectadasHog, pontuacao, idx = detectorHog.run(imagem, 2) #metodo run retorna parametros como valor da confianca
                                                                #aumenta a escala em 2 vezes

detectorCNN = dlib.cnn_face_detection_model_v1("recursos/mmod_human_face_detector.dat") #arquivo que já foi treinado
facesDetectadasCNN = detectorCNN(imagem, 2) #mesma resolução para teste

for i, d in enumerate(facesDetectadasHog):  #percorrendo bounding boxes do HOG
    print(pontuacao[i])
print("")
for face in facesDetectadasCNN:
    print(face.confidence)

