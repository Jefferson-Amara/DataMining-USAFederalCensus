# -*- coding: utf-8 -*-
"""
Mineração de Dados - Trabalho 5
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing

#Carregando arquivos de dados
data = pd.read_csv("censo-dados-07.csv")
atrib = pd.read_table("nomes-das-colunas.dat",sep="\t",header=None)
atrib = pd.DataFrame(atrib.iloc[:,0])
possible = pd.read_table("valores-possiveis.dat",sep=":",header=None)
#Substituindo cabeçalhos originais das colunas pelos atributos
data.columns = atrib.values.T.tolist()

#Definindo atributos categoricos
for i in data.columns:
    if data[i].dtypes == object:
        data[i] = data[i].astype('category')

#Selecionando atributos numericos
dados_numericos = data._get_numeric_data()
n = pd.DataFrame(dados_numericos.columns.values)
#Normalizando dados numericos
min_max_scaler = preprocessing.MinMaxScaler()
numeric_scaled = min_max_scaler.fit_transform(dados_numericos)
dados_numericos = pd.DataFrame(numeric_scaled)
dados_numericos.columns = n.values.T.tolist()       
        
#Substituindo insconsistencias   
#data = data.replace( ' ?','-999' )

#Concatenando dados categoricos e numericos apos pre-processamento
dados_categoricos = data.select_dtypes(exclude=[np.number])
data = pd.concat([dados_categoricos,dados_numericos],axis=1)

#Liberando variáveis temporárias
del atrib, i, n, numeric_scaled


