import os
import cv2
from PIL import Image
import numpy as np

import csv
import pickle

from scipy.spatial.distance import cosine
from deepface import DeepFace

from mtcnn.mtcnn import MTCNN
detector = MTCNN()                                      # detecção de faces na imagem

#################################################################################################################
def match_face(emb_buscado, database, min_distance=0.50):
    # embedding da imagem pesquisada
    # database é um DICIONÁRIO no formato:
    #   ID : {nome_da_pessoa : embedding}
    '''The function `match_face` takes an input face embedding and a database of face embeddings, and
    returns the name of the closest matching face in the database if the distance is below a certain
    threshold, otherwise it returns None.
    
    Parameters
    ----------
    embedding
        The embedding parameter is a numerical representation of a face. It is typically a vector of
    numbers that captures the unique features of a face.
    database
        The database parameter is a dictionary that contains the embeddings of known faces. The keys of the
    dictionary are the names of the individuals and the values are the corresponding embeddings.
    '''
    # obs.: optou-se por utilizar para validação dos embeddings o cálculo através da função cosine()
    #       pois a outra opção era treinar uma rede de classificação a partir da imagens das fotos de cada pessoa
    #       Este método de classificação, entretanto, exige um novo treinamento do modelo toda vez que uma nova
    #       pessoa for incluída no cadastro, o que é lento.
    #       Desta forma utilizando a função cosine() não é necessário mais fazer o treinamento da rede neural mas
    #       apenas acrescentar o embedding das imagens da nova pessoa no arquivo de cadastro das imagens através
    #       dos códigos:
    #           rfacial_prepara.py -> extrai as imagens de faces de uma foto
    #           Obs.: manualmente faz-se uma verificação para exclusão de alguma face indevida dos arquivos
    #           rfacial_prepara_emb.py -> gera o embedding das imagens e coloca no arquivo de cadastro

    min_distance = 0.25  # Os valores calculados pela função cosine() variam entre 0 e 1 sendo quão mais próximo
                        # de 0 indicando uma melhor correspondência


    # Loop over all faces in the database
    for aux in sorted(database):
        db_embedding = database[aux]

        distance = cosine(emb_buscado[0]['embedding'], db_embedding[0]['embedding'])        

        # If the distance is less than the min_distance, update
        # the min_distance and match
        if distance < min_distance:
            aux = aux.split(' - ')[-1]
            return aux
    return None
    
def extracao_face(image, box, required_size=(160,160)):
    # transforma a imagem recebida para o formato do numpy (melhor performance)
    img = np.asarray(image)
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extração da face 
    pixels = img[y1:y2, x1:x2]
    # criam uma imagem (imagem) a partir de uma array (pixels)
    imagem = Image.fromarray(pixels)
    imagem = imagem.resize(required_size)
    return np.asarray(imagem)    
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
'''
arquivo2 = "C:\\BACKUP\\Cursos\\IA\\yolov8\\Dados\\rfacial\\prepara\\Angelina Jolie\\039_6b10ab98.jpg"
arquivo = "C:\\BACKUP\\Cursos\\IA\\yolov8\\Dados\\rfacial\\cadastro\\Angelina Jolie\\foto-Angelina Jolie-1.jpg"    
emb1 = DeepFace.represent(arquivo, model_name="Facenet", enforce_detection=False)
print(emb1)
emb2 = DeepFace.represent(arquivo2, model_name="Facenet", enforce_detection=False)
print(emb2)
#distance3 = DeepFace.find(emb1[0]["embedding"], emb2[0]["embedding"], distance_metric = 'cosine', model_name="Facenet")
distance3 = cosine(emb1[0]["embedding"],emb2[0]["embedding"])
print(f'Distancia calculada: {distance3}')
exit()
'''

arq_cadastro = './dados/rfacial/cadastro/pessoas_cadastradas.pkl'
cadastro_pessoas = {}
try:
    with open(arq_cadastro, "rb") as arquivo:    
        #aux_cadastro_pessoas = pickle.load(arquivo)
        aux = pickle.load(arquivo)
        cadastro_pessoas = dict(aux)
        print(f'Aberto arquivo com dados das pessoas cadastradas...{len(cadastro_pessoas)} registros')
        #print(list(cadastro_pessoas.keys()))
        # os dados de cadastro são armazenados em um dicionário no formato:
        #   identificador : embedded
        # Note que o identificador tem o formato número com 8 algarismos seguido de "-" e o nome da pessoa
        # Com isso ao ordenarmos em ordem alfabética primeiro faremos a busca em 01 face de cada pessoa antes de
        # tentar outra face cadastrada da mesma pessoa. Com isso evitamos de passar várias vezes por uma mesma pessoa
        # antes de testarmos as demais...
except:
    print(f'Não consegui abrir o arquivo.')
    exit()

backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn','retinaface', 'mediapipe','yolov8','yunet','centerface',]
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",]    
metrics = ["cosine", "euclidean", "euclidean_l2"]


arquivo = "C:\\BACKUP\\Cursos\\IA\\yolov8\\Dados\\rfacial\\teste2.jpeg"

imagem = cv2.imread(arquivo)
escolha_extrator = "mtcnn"
#escolha_extrator = "deepface"
if escolha_extrator == "deepface":
    face_objs = DeepFace.extract_faces(img_path = arquivo, detector_backend = backends[3])
    #face_objs = DeepFace.extract_faces(imagem, detector_backend = backends[3])
else:
    face_objs = detector.detect_faces(imagem)

print(f'Qtde encontrada: {len(face_objs)}')

cor_caixa_BGR = (0,0,255)
espessura_linha = 2
fonte_caracter = cv2.FONT_HERSHEY_SIMPLEX
fonte_escala = 0.5
face_embedding = None
#cv2.imshow('Entrada', imagem)
#cv2.waitKey(0)

titulo = ''

for face in face_objs:
    if escolha_extrator == "deepface":
        # converte de RGB para BGR
        img = face['face'][:,:,::-1]      
        cv2.imshow('entrada', img)
        cv2.waitKey(0)  
        x1 = face['facial_area']['x']
        y1 = face['facial_area']['y']
        w = face['facial_area']['w']
        h = face['facial_area']['h']    
    else:
        confidence = face["confidence"]
        if confidence < 0.98:
            continue
        x1,y1,w,h = face["box"]
        img = extracao_face(imagem, face["box"])
    print('Fazendo a codificação da face (embedding)...')

    #if face_embedding[0]['face_confidence'] == 0:
    #    continue
    
    # curiosamente nem todas as faces encontradas pelo próprio DeepFace este reconhece como válidas para fazer o embedding
    # Desta forma é necessário ajustar enforce_detection=False para ignorar estes casos...
    face_embedding = DeepFace.represent(img, model_name="Facenet", enforce_detection=False)

    identificado = match_face(face_embedding, cadastro_pessoas)

    if identificado == None:
        titulo = 'desconhecido'
        cor_caixa_BGR = (0,0,255)
    else:
        titulo = identificado
        cor_caixa_BGR = (0,255,0)

    cv2.rectangle(imagem, (x1,y1), (x1+w,y1+h), cor_caixa_BGR, espessura_linha)
    cv2.putText(imagem, titulo, (x1,y1-10), fonte_caracter, fontScale= fonte_escala, color= cor_caixa_BGR, thickness= 1)
    print(f'Resultado da pesquisa: {titulo}')

#cv2.imshow('Reconhecimento Facial', imagem)
import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(7, 4)
plt.axis("off")
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
plt.show()
cv2.waitKey(0)

