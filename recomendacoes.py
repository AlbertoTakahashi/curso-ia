pip install annoy

import tensorflow as tf
print(tf.__version__)
device_name = tf.test.gpu_device_name()

import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.utils import image_dataset_from_directory, split_dataset
from tensorflow.data.experimental import cardinality

import numpy as np
import os
import pickle
import annoy

from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

os.chdir('./drive/MyDrive')
path_dados = './base_treinamento'       
path_dados_teste = './base_testes'      
dir_saida = './checkpoint'             
checkpoint_file = '/tenis_chinelo_model.keras'                     
arquivo_classes = '/classes.txt'                                    
extrator_caracteristicas = '/extrator_caracteristicas.model.keras'  
cadastro_imagens = '/imagens_processadas'
arq_indice = '/indice'             
arq_dicionario = '/dicImagens'     
image_size = [224, 224]  
batch_size = 50                
info_diretorio = [x for x in os.walk(path_dados)]
sub_diretorios = info_diretorio[0][1]
lista_de_classes = sub_diretorios                  

def gera_indice(dados, dir_saida, arq_ind, arq_dic):
    if len(dados) == 0:
        print('Faltaram os dados de entrada...')
        exit()
    contador = 0
    dArquivosImagem = {}
    qtde_features = len(list(dados.values())[0])
    t = annoy.AnnoyIndex(qtde_features, 'angular')              
    for nome_arquivo, array_caracteristicas in dados.items():
        contador += 1
        t.add_item(contador, array_caracteristicas)             
        dArquivosImagem.update({contador:nome_arquivo})         
    t.build(10)    
    arq_saida = dir_saida + arq_ind + '.ann'
    t.save(arq_saida)                                             
    arq_saida = dir_saida + arq_dic + '.pkl'
    with open(arq_saida, 'wb') as file:
        pickle.dump(dArquivosImagem, file)  
                                                                    
treinamento = False
classificacao_ou_extratorCaracteristicas = "recomendacao"
if treinamento:
    if classificacao_ou_extratorCaracteristicas == "classificacao":        
        pass #excluido pois não está no contexto deste exercício

    elif classificacao_ou_extratorCaracteristicas == "extratorCaracteristicas":
        model = VGG16(input_shape= image_size + [3], weights='imagenet', include_top=False)
        x = model.output
        x = GlobalAveragePooling2D()(x)     
        model = Model(inputs=model.input, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
        model.save(dir_saida + extrator_caracteristicas)
    else:
      print('não foi selecionada nenhuma opção válida')
      exit()
else:      
  salva_so_pesos = False 
  if classificacao_ou_extratorCaracteristicas == "classificacao":
    model = keras.models.load_model(dir_saida + checkpoint_file)
    lista_de_classes = []
    try:
      with open(dir_saida + arquivo_classes, "r") as arquivo:
        for linha in arquivo:
          lista_de_classes.append(linha)
    except Exception as exc:
      print(f'Não consegui escrever o arquivo. Olha o erro: {exc}')
      exit()
  elif classificacao_ou_extratorCaracteristicas == "extratorCaracteristicas":
    model = keras.models.load_model(dir_saida + extrator_caracteristicas)
  elif classificacao_ou_extratorCaracteristicas == "recomendacao":
    model = keras.models.load_model(dir_saida + extrator_caracteristicas)   
  else:
    print('não foi selecionada nenhuma opção válida')
    exit()    
  if classificacao_ou_extratorCaracteristicas == "classificacao":
    img_path = 'imagem_teste_tenis.jpeg'
    img = keras.utils.load_img(img_path, target_size=image_size)
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
  elif classificacao_ou_extratorCaracteristicas == "extratorCaracteristicas":
    cadastro = {}
    totalizador = 0
    contador = 0
    info_diretorio = [x for x in os.walk(path_dados)]
    sub_diretorios = info_diretorio[0][1]
    qtde_subdiretorios = len(sub_diretorios)
    for controle_subdir in range(1,qtde_subdiretorios+1):
      este_diretorio = info_diretorio[controle_subdir][0]
      quantos_arquivos_imagem = len(info_diretorio[controle_subdir][2])
      contador = 0
      for arquivo in info_diretorio[controle_subdir][2]:
        contador += 1
        arquivo_path = este_diretorio + '/' + arquivo
        img = keras.utils.load_img(arquivo_path, target_size=image_size)
        x = keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        pred = model.predict(x)
        cadastro.update({arquivo_path : pred[0]})
      totalizador += contador
    try:
      with open(dir_saida + cadastro_imagens, "wb") as arquivo:
            pickle.dump(cadastro, arquivo)
    except Exception as exc:
      print(f'Não consegui escrever o arquivo. Olha o erro: {exc}')
    gera_indice(cadastro, dir_saida, arq_indice, arq_dicionario)    
  elif classificacao_ou_extratorCaracteristicas == "recomendacao":
    cadastro = {}
    try:
        with open(dir_saida + cadastro_imagens + '.pkl', "rb") as arquivo:    
            cadastro = pickle.load(arquivo)
        qtde_features = len(list(cadastro.values())[0])
    except Exception as exc:
        print(f'Não consegui abrir o arquivo. Olha o erro: {exc}')
        exit() 
    indice_annoy = annoy.AnnoyIndex(qtde_features, 'angular')
    indice_annoy.load(dir_saida + arq_indice + ".ann")
    try:
        with open(dir_saida + arq_dicionario + '.pkl', "rb") as arquivo:    
            dArquivosImagem = pickle.load(arquivo)
    except:
        print(f'Não consegui abrir o arquivo de dicionario. Olha o erro: {exc}')
    img_path = './imagem_teste_tenis.jpeg'
    img = keras.utils.load_img(img_path, target_size=image_size)
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)[0] 
    quantos_vizinhos = 4
    recomendacoes = indice_annoy.get_nns_by_vector(pred, quantos_vizinhos)
    for aux in recomendacoes:
    import matplotlib.pylab as plt
    import matplotlib.image as mpimg
    plt.figure(figsize=(3, 3))
    plt.title('Imagem de referência')
    plt.imshow(mpimg.imread(img_path))
    plt.axis('off')
    plt.tight_layout()
    linhas = 1
    colunas = quantos_vizinhos
    indice_subplot = 0
    plt.subplots(linhas, colunas, figsize=(15, 3),layout='constrained')
    for aux in recomendacoes:
      indice_subplot += 1
      plt.subplot(linhas,colunas, indice_subplot)
      plt.title(dArquivosImagem[aux])
      plt.axis('off')
      plt.imshow(mpimg.imread(dArquivosImagem[aux]))
    plt.show()      
