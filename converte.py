import cv2
import numpy as np
imagem_colorida = cv2.imread('Lenna_color.png')
dimensoes = imagem_colorida.shape
print(dimensoes)
size_x, size_y, canais = dimensoes
imagem_cinza = np.zeros((size_x,size_y,1),dtype=np.uint8)
imagem_preto_branco = np.zeros((size_x,size_y,1),dtype=np.uint8)
# cinza = 0.2126R+0.7152G+0.0722B
for x in range(size_x):
    for y in range(size_y):
        imagem_cinza[x,y] = imagem_colorida[x,y,0] * 0.2126 + imagem_colorida[x,y,1] * 0.7152 + imagem_colorida[x,y,2] * 0.0722
        imagem_preto_branco[x,y] = 0 if imagem_cinza[x,y] < 128 else 255
print(imagem_cinza.shape)
cv2.imshow('Colorido', imagem_colorida)
cv2.imshow('Cinza', imagem_cinza)
cv2.imshow('Preto e Branco', imagem_preto_branco)
cv2.waitKey(0)