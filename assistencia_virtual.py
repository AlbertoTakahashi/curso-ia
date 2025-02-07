##############################################################
# Teste utilizando facilidades text-to-speech e speech-to-text
##############################################################
import speech_recognition as sr
from gtts import gTTS
import datetime
import os
from playsound3 import playsound

import requests
import json

rec = sr.Recognizer()
#print(sr.Microphone().list_microphone_names())
# lista todos os microfones instalados no computador

def falar(texto):
    print(f'cheguei para falar: {texto}')
    #falante = gTTS(text=texto, lang='pt-BR', slow=True)
    falante = gTTS(text=texto, lang='pt-BR')
    nome_arquivo = "voice.mp3"
    falante.save(nome_arquivo)
    playsound(nome_arquivo)
    try:
        os.remove(nome_arquivo)
    except OSError:
        pass

def horas_pra_texto(horas):
    hora = horas.hour
    minutos = horas.minute
    texto = "Agora são " + str(hora)
    if hora > 1:
        texto += " horas"
    else:
        texto += " hora"
    if minutos > 0:
        if minutos < 2:
            texto += " e 1 minuto"
        else:
            texto += " e " + str(minutos) + " minutos"
    return texto

def hgbrasil():
    # API para consulta de dados metereológicos
    Curitiba = 455822
    URL_pesquisa = "https://api.hgbrasil.com/weather?woeid=" + str(Curitiba)
    retorno = requests.request("GET", URL_pesquisa)
    resposta = json.loads(retorno.text)
    #print(resposta) 
    temperatura = resposta['results']['temp']
    descricao = resposta['results']['description']
    cidade = resposta['results']['city_name']
    return(cidade,temperatura,descricao)
     
def clima_pra_texto(cidade,temperatura,descricao):
    texto = "Agora " + cidade + " está com " + descricao + " e a temperatura é de " + str(temperatura) + " graus"
    #print(texto)
    return texto
################################################################################################

sair = False

with sr.Microphone(device_index = 1) as microfone:
    rec.adjust_for_ambient_noise(microfone)
    print('Tente falar algo. Estou ouvindo...')
    while not sair:
        try:
            print("ainda estou aqui...")
            #audio = rec.listen(microfone,2,2)
            audio = rec.listen(microfone,2)
            print("Agora ouvi.")
            texto = rec.recognize_google(audio, language='pt-BR')
            if ("chega" in texto):
                sair = True
                print()
                print("Ok. Até a próxima.")
                print("Fui...")
            else:
                print(f'Olha o que eu entendi do que você disse: "{texto}"')
                print()
                if "hora" in texto:
                    falar(horas_pra_texto(datetime.datetime.now()))
                elif "tempo" in texto:
                    cidade,temperatura,descricao = hgbrasil()
                    falar(clima_pra_texto(cidade,temperatura,descricao))
                    
        except:
            #print()
            #print("Algo deu errado...")
            #print("Vamos tentar denovo!")
            #print()
            pass
