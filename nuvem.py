# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 13:02:00 2017

@author: fabiopsan
"""


# -*- coding: utf-8 -*-
#Conjunto de importações
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd

print("Armazenamento de documentos (frases) na lista dataset")

dados = pd.read_csv("c:\\tese\\pesquisa100.csv", sep=",", encoding="utf-8")
							                  
print("Armazenamento das polaridades de cada documento (frase) na lista polaris")
dataset = dados["OPINIAO"]
polaris = dados["POLARIDADE"]
						        
print("Divisão dos dados das listas dataset e polaris em conjuntos de treinamento e validação")
dados_treino, dados_val, pols_treino, pols_val = train_test_split(dataset, polaris, test_size=0.30)

       
from nltk.corpus import stopwords
portuguese_stops = set(stopwords.words('portuguese'))

portstop = nltk.corpus.stopwords.words('portuguese')
content = [w for w in dataset if w.lower() not in portstop]
x = len(content) / len(dataset)
print(x)

#conteudo = [palavra for palavra in dataset if palavra.lower() not in portuguese_stops]
conteudo = [palavra for palavra in portuguese_stops if palavra.lower() not in dataset ]

print(len(conteudo)/len(dataset))
print(len(conteudo))
print( len(dataset))
print(conteudo)

#print(stopwords.words('portuguese'))
h=len(stopwords.words('portuguese'))
print(h)

#tokenização
from nltk.tokenize import word_tokenize
#print(word_tokenize(dataset))
  
#stemizacao
from nltk.stem import SnowballStemmer
port_stem = SnowballStemmer('portuguese')
print(port_stem.stem(dataset))

print(type(dataset))



print("\n---------------------------------------------\n")
#Print do conjunto de validação e suas respectivas polaridades
print("Conjunto de Validação")
print(dados_val)

print("Polaridades do Conjunto de Validação")
print(pols_val)

print("Cria uma instância para a bag-of-words" )  
bag = CountVectorizer()
print(bag)

print(" Método fit_transform:fit = cria e aprende a bag E transform = cria a matriz termo-documento")
bag_treino = bag.fit_transform(dados_treino)
print(" Printa o vocabulário da bag-of-words")   

#A função sorted() ordena o vocabulário da bag-of-words   
print(sorted(bag.vocabulary_))
pausa = input("Pausa..." )

#Printa a bag-of-words    
print("bag treino  --- bag of words" )  
print(bag_treino)

#print(bag_treino.__ror__)
pausa = input("Pausa..." )

#Cria a matriz termo-documento para o conjunto de validação com a bag já treinada
bag_val = bag.transform(dados_val)
print(" Imprime a matriz termo-documento criada para o conjunto de validação")    
print(bag_val)


print("Cria uma instância para o algoritmo Multinomial Naive Bayes")  
nb_modelo = MultinomialNB()
print(nb_modelo)

print("O método fit treina o modelo utilizando o algoritmo Multinomial Naive Bayes")
print("O argumento da bagofwords deve ser passado no formato array")
pausa = input("Pausa...")
nb_modelo.fit(bag_treino.toarray(), pols_treino)
print(bag_treino.toarray())


print("Realiza as predições para o conjunto de treinamento")
pols_pred_treino = nb_modelo.predict(bag_treino.toarray())
print("Realiza as predições para o conjunto de validação")
pols_pred_val = nb_modelo.predict(bag_val.toarray())
print("Imprime as predições calculadas para ambos os conjuntos")
print("Polaridades preditas para o conjunto de treinamento")
print(pols_pred_treino)
print("Polaridades preditas para o conjunto de validação")
print(pols_pred_val)

print("Calcula a acurácia das predições realizadas para o conjunto de treinamento")
print("Acurácia no treinamento")
print(accuracy_score(pols_treino, pols_pred_treino))
print("Calcula a acurácia das predições realizadas para o conjunto de validação")
print("Acurácia na validação")
print(accuracy_score(pols_val, pols_pred_val))
print("Acurácia no treinamento")
print("Com o argumento 'normalize=False' o resultado da acurácia é o total de acertos calculados")
print( accuracy_score(pols_treino, pols_pred_treino, normalize=False))
print("****Acurácia na validação ***")
print(accuracy_score(pols_val, pols_pred_val, normalize=False))


print("Armazenamento da frase de teste na variável frase_teste")
frase_teste = [("o professor é muito bom para ensinar")]

print("Cria a bag-of-words para a frase_teste")
bag_teste = bag.transform(frase_teste)

print("Aplica o modelo Multinomial Naive Bayes aprendido na bag criada")
pol_pred_teste = nb_modelo.predict(bag_teste.toarray())
                
print("Estrutura de decisão para apresentar o resultado como String")

#Resultado = 1 ==> Polaridade POSITIVO
#Resultado = -1 ==> Polaridade NEGATIVO

if pol_pred_teste == 1:
   print("POSITIVO")
else:
   print("NEGATIVO")

#Matriz de confusão para os resultados da validação
print("Matriz de Confusão")
print(confusion_matrix(pols_val, pols_pred_val))

