import cv2
import dlib

#cinco subdetectores utilizados pelo dlib
subdetector = ["face frontal", "face a esquerda", "face a direita", "a frente girando a esquerda", "a frente girando a direita"]

imagem = cv2.imread("fotos/grupo.0.jpg")
detector = dlib.get_frontal_face_detector() #nao precisa passar os haar cascades como parametro
facesDetectadas, pontuacao, idx = detector.run(imagem, 1, -1)   #faces detectadas
                                                                #confiabilidade da classificacao
                                                                #idx indica qual subdetector foi utilizado (face frontal ou lateral)

#print(facesDetectadas)
#print(pontuacao)
#print(idx)

for i, d in enumerate(facesDetectadas):
    #print(i)
    #print(d)
    print("deteccao: {}, pontuacao: {}, subdetector: {}".format(d, pontuacao[i], subdetector[int(idx[i])]))
    e, t, d, b = (int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))
    cv2.rectangle(imagem, (e,t), (d, b), (0,0,255), 2)

cv2.imshow("detector hog", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()