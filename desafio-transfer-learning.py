#%matplotlib inline
#import tensorflow
#import keras

import os
import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from pathlib import Path

dir_atual = (Path(__file__).parent) 
dir_raiz = Path(__file__).parent.parent
dir_dados = str(dir_raiz) + '\\dados\\caltech-101\\101_ObjectCategories'
os.chdir(dir_dados)
#root = os.getcwd() 
#print(root)

exclude = ['BACKGROUND_Google', 'Motorbikes', 'airplanes', 'Faces_easy', 'Faces']
train_split, val_split = 0.7, 0.15

categories = [x[0] for x in os.walk(dir_dados) if x[0]][1:]
categories = [c for c in categories if c not in [os.path.join(dir_dados, e) for e in exclude]]

print(categories)


from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
#from keras import load_img, img_to_array
from tensorflow.keras.utils import load_img, img_to_array
print('oi eu aqui')


# helper function to load image and return it and input vector
def get_image(path):
    # carrega uma imagem e a redimensiona para (224, 224)
    img = load_img(path, target_size=(224, 224))
    # Converts a PIL Image instance to a NumPy array
    x = img_to_array(img)
    #print(f'Dimensões originais: (tamanho_X x tamanho_Y x 3_canais_cor_RGB) {x.shape}')
    # aumenta as dimensões da imagem
    x = np.expand_dims(x, axis=0)
    #print(f'Dimensões expandido: (1 x tamanho_X x tamanho_Y x 3_canais_cor_RGB) {x.shape}')
    # Obs.: essa dimensão adicional será utilizada quando fizermos a agregação das diferentes
    #       imagens em um único vetor: (samples, size1, size2, channels)

    # Pré-processa um tensor ou uma matriz Numpy que codifica um lote de imagens
    # A função preprocess_input() tem como objetivo adequar sua imagem ao formato que o modelo requer.
    # Alguns modelos usam imagens com valores que variam de 0 a 1. Outros de -1 a +1. Outros usam o 
    # estilo "caffe", que não é normalizado, mas é centralizado.
    x = preprocess_input(x)
    # def _preprocess_numpy_input(x, data_format, mode):
    # Preprocesses a Numpy array encoding a batch of images.
    # Arguments:
    #  x: Input array, 3D or 4D.
    #  data_format: Data format of the image array.
    #  mode: One of "caffe", "tf" or "torch".
    #      - caffe: will convert the images from RGB to BGR,
    #          then will zero-center each color channel with
    #          respect to the ImageNet dataset,
    #          without scaling.
    #      - tf: will scale pixels between -1 and 1,
    #          sample-wise.
    #      - torch: will scale pixels between 0 and 1 and then
    #          will normalize each channel with respect to the
    #          ImageNet dataset.
    # "mode" se for omitido significa que a configuração global keras.backend.image_data_format()
    # é usada -> usará caffe (a menos que você a tenha alterado, ela usa "channels_last"). O padrão é None.
    #
    # Returns:
    #  Preprocessed Numpy array.
    return img, x

# Load all the images from root folder
data = []
print(categories)
# enumerate() adiciona um contador (no caso este retorna na variável "c") para cada elemento 
# da coleção "categories"
# Ou seja, "c" conterá um valor sequencial equivalente a posição de cada elemento retirado da coleção / lista
# "categories"
for c, category in enumerate(categories):
    print('oi eu aqui 2')
    print(c)
    print(category)
    images = [os.path.join(dp, f) for dp, dn, filenames 
              in os.walk(category) for f in filenames 
              if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    #print(images)
    for img_path in images:
        print(img_path)
        img, x = get_image(img_path)
        # img contém a imagem original convertida para o tamanho 224 x 224
        # x[0] contém a imagem convertida / normalizada
        # "c" contém um número sequencial para cada categoria de imagem processada
        # Ou seja:
        #  x : imagem,  y : contador_sequencial_para_cada_categoria_de_imagens_processadas
        data.append({'x':np.array(x[0]), 'y':c})
    #print(data)

# count the number of classes
num_classes = len(categories)
print(f'Numero de classes lidas: {num_classes}')

# Randomize (embaralha) the data order.
random.shuffle(data)

# create training / validation / test split (70%, 15%, 15%)
train_split = 0.7
val_split = 0.15
idx_val = int(train_split * len(data))
idx_test = int((train_split + val_split) * len(data))
train = data[:idx_val]
val = data[idx_val:idx_test]
test = data[idx_test:]
print(f'Tamanho da base de treinamento: {len(train)}')
print(f'Tamanho da base de validação: {len(val)}')
print(f'Tamanho da base de testes: {len(test)}')
print()
#print('Base de validação:')
#print(test)

# Separate data for labels.
# lembrando que o formato de cada um dos vetores é:
#  x : imagem,  y : contador_sequencial_para_cada_categoria_de_imagens_processadas
# Então teremos: x_train -> vetor com as imagens de treinamento
#                y_train -> vetor com as identificações numéricas de cada categoria que compõe a base
x_train, y_train = np.array([t["x"] for t in train]), [t["y"] for t in train]
x_val, y_val = np.array([t["x"] for t in val]), [t["y"] for t in val]
x_test, y_test = np.array([t["x"] for t in test]), [t["y"] for t in test]
#print(x_test, y_test)

#Pre-process the data as before by making sure it's float32 and normalized between 0 and 1.
# Let's get a summary of what we have.

# normalize data
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# convert labels to one-hot vectors
from keras.utils import to_categorical
# a função to_categorical() transforma um valor para o formato "one hot"
# Ou seja, por exemplo, supondo um conjunto [1,3,4,2,1]
#   inicialmente identificará qual o maior valor -> 4
#   depois converterá cada valor para o formato "one hot" de modo que esta lista passará para
#       [
#           [0 1 0 0 0]
#           [0 0 0 1 0]
#           [0 0 0 0 1]
#           [0 0 1 0 0]
#           [0 1 0 0 0]
#       ]
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)
#print(y_test)
#print(y_test.shape)

# summary
print("finished loading %d images from %d categories"%(len(data), num_classes))
print("train / validation / test split: %d, %d, %d"%(len(x_train), len(x_val), len(x_test)))
print("training data shape: ", x_train.shape)
print("training labels shape: ", y_train.shape)

#images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dir_dados) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
#idx = [int(len(images) * random.random()) for i in range(8)]
#imgs = [image.load_img(images[i], target_size=(224, 224)) for i in idx]
#concat_image = np.concatenate([np.asarray(img) for img in imgs], axis=1)
#plt.figure(figsize=(16,4))
#plt.imshow(concat_image)
#plt.show()

# build the network
model = Sequential()
print("Input dimensions: ",x_train.shape[1:])

model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

# compile the model to use categorical cross-entropy loss function and adadelta optimizer
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_train[0])
print(y_train[0])

history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=2,
                    validation_data=(x_val, y_val))

fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
print(history.history["val_loss"])
ax.plot(history.history["val_loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
print(history.history["val_accuracy"])
ax2.plot(history.history["val_accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

print('Só falta salvar')
model.save("calltech-101-animais-50epoch.keras")
print('kbo...')
exit()
