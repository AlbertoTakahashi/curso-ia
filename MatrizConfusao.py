!pip install -q tensorflow-gpu==2.0.0-beta1

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np

import seaborn as sns

import pandas as pd

tf.__version__

%load_ext tensorboard

logdir='log'

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images, test_images = train_images / 255.0, test_images / 255.0

classes=[0,1,2,3,4,5,6,7,8,9]

print(train_images.shape)
print(test_images.shape)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=train_images,
            y=train_labels,
            epochs=5,
            validation_data=(test_images, test_labels))

y_true=test_labels
predict_y=model.predict(test_images) 
y_pred=np.argmax(predict_y,axis=1)

classes=[0,1,2,3,4,5,6,7,8,9]

conf_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()

soma_colunas = np.sum(conf_mat, axis=0)
soma_linhas = np.sum(conf_mat, axis=1)
tamanho_base_dados = np.sum(conf_mat)
# verdadeiro positivo
VP = np.diag(conf_mat)
# falso negativo
FN = soma_linhas - VP
# falso positivo
FP = soma_colunas - VP
# verdadeiro negativo
VN = tamanho_base_dados - (FN + FP + VP)
print(conf_mat)
print()
print('Verdadeiro Positivo:')
print(VP)
print()
print('Falso Negativo:')
print(FN)
print()
print('Falso Positivo:')
print(FP)
print()
print('Verdadeiro Negativo:')
print(VN) 
# sensibilidade = VP / (VP + FN)
# especificidade = VN / (FP + VN)
# acurácia = (VP + VN) / (VP + VN + FP + FN)
# precisão = VP / (VP + FP)
# F-Score = 2 x (precisao x sensibilidade) / (precisao + sensibilidade)
sensibilidade = VP / (VP + FN)
especificidade = VN / (FP + VN)
acuracia = (VP + VN) / (VP + VN + FP + FN)
precisao = VP / (VP + FP)
f_score = 2 * (precisao * sensibilidade) / (precisao + sensibilidade)
print()
print('Sensibilidade:')
print(sensibilidade)
print()
print('Especificidade:')
print(especificidade)
print()
print('Acurácia:')
print(acuracia)
print()
print('Precisão:')
print(precisao)
print()
print('F-Score:')
print(f_score)

con_mat_norm = np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], decimals=2)

con_mat_df = pd.DataFrame(con_mat_norm,
                     index = classes,
                     columns = classes)

figure = plt.figure(figsize=(4, 4))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

