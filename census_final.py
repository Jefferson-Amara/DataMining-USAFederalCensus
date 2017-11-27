# -*- coding: utf-8 -*-
"""
Mineração de Dados - Trabalho 5
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as its

from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Configurando avisos do Pandas
pd.options.mode.chained_assignment = None

def load_data(data_file_path, att_file_path, isTestData=False):
    # Carregando arquivo de dados
    data = pd.read_csv(data_file_path)
    att = pd.read_table(att_file_path, sep='\t', header=None)
    
    # Verifica se é dado de teste, caso seja, não há o atributo 'target' nos dados
    if isTestData:
        att = att.iloc[0:(len(att) - 1), 0]
    else:
        att = att.iloc[:, 0]
    
    # Definindo nome dos atributos
    data.columns = att.values.T.tolist()
    
    return data

def pre_process_data(data, isTestData=False):
    # Substituindo valores numéricos int64 por float64
    numeric_data = data.select_dtypes(["int64"]).columns
    data[numeric_data] = data[numeric_data].astype(float)
    
    if not isTestData:
        # Removendo outliers nos atributos numéricos
        data = remove_outlier(data)
    
    # Substituindo dados do tipo object por category
    for i in data.columns:
        if data[i].dtypes == object:
            data[i] = data[i].astype("category")
            
    # Selecionando atributos categóricos
    category_data = data.select_dtypes(["category"]).columns
    
    # Substituindo valores ausentes nos atributos categóricos
    data[category_data] = data[category_data].replace(' ?', 'NaN')
    
    # Substituindo dados do tipo object por category
    for i in data.columns:
        if data[i].dtypes == object:
            data[i] = data[i].astype("category")

    if not isTestData:
        # Removendo atributos categóricos que possuam valores ausentes acima de 50%
        for i in category_data:
            if len(data[data[i] == 'NaN']) >= (0.5 * len(data[i])):
                del data[i]
    
        # Selecionando atributos categóricos
        category_data = data.select_dtypes(["category"]).columns
    
    # Substituindo os dados categóricos por numéricos
    data[category_data] = data[category_data].apply(lambda x: x.cat.codes)
    
    # Alterando os dados para float64
    data[data.columns] = data[data.columns].astype(float)

    # Normalizando os dados numericos
    min_max_scaler = preprocessing.MinMaxScaler()
    numeric_data_scaled = min_max_scaler.fit_transform(data)
    data_scaled = pd.DataFrame(numeric_data_scaled)
    
    data_columns = data.columns
    data_scaled.columns = data_columns

    return data_scaled

def remove_outlier(data):
    # Removendo outliers nos atributos numéricos
    numeric_data = data.select_dtypes(["float64"]).columns
    
    for i in numeric_data:
        # Removendo outliers nos dados numéricos
        q1, q3 = np.percentile(data[i], [25 ,75])
        iqr = q3 - q1
        min_val = q1 - (iqr * 1.5)
        max_val = q3 + (iqr * 1.5)
        
        data[i] = data[(data[i] >= min_val) & (data[i] <= max_val)]
        
    data = data.dropna()
    
    return data

def oversampling_data(data):
    # Separando as classes
    data_under_50000 = data[data["income level"] == 0]
    data_over_50000 = data[data["income level"] == 1]
     
    # Oversampling
    data_over_50000_oversampled = resample(data_over_50000, replace=True, n_samples=len(data_under_50000))
     
    # Combina as classes equilibradas
    data_oversampled = pd.concat([data_over_50000_oversampled, data_under_50000])
    
    return data_oversampled

def undersampling_data(data):
    # Separando as classes
    data_under_50000 = data[data["income level"] == 0]
    data_over_50000 = data[data["income level"] == 1]
     
    # Undersampling
    data_under_50000_undersampled = resample(data_under_50000, replace=False, n_samples=len(data_over_50000))
     
    # Combina as classes equilibradas
    data_undersampled = pd.concat([data_under_50000_undersampled, data_over_50000])
    
    return data_undersampled

def draw_conf_matrix(conf_mat):
    conf_mat = conf_mat.astype("float")
    conf_mat = conf_mat / conf_mat.sum(axis=1)[:, np.newaxis]
    
    cmap=plt.cm.Blues
    plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    iclasses = np.arange(2)
    
    plt.xticks(iclasses, ['-50000', '50000+'], rotation=45)
    plt.yticks(iclasses, ['-50000', '50000+'])
    
    limiarCor = conf_mat.max() / 2.
    
    for i, j in its.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > limiarCor else "black")
        
    plt.tight_layout()
    plt.ylabel("Valores Esperados")
    plt.xlabel("Valores Preditos")
    plt.show()
    
def draw_roc_curve(model_name_arr, color_arr, fpr_dt, tpr_dt, auc_dt):
    for i in range(0, len(model_name_arr)):
        plt.plot(fpr_dt[model_name_arr[i]].values[0], 
                 tpr_dt[model_name_arr[i]].values[0], 
                 color=color_arr[i], lw=2, 
                 label=model_name_arr[i] + " Area = %0.2f" % np.mean(auc_dt[model_name_arr[i]]))
        
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("Falso Positivo")
    plt.ylabel("Verdadeiro Positivo")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()
    
def do_roc_curve(model, test_x, test_y):
    predict_prob = model.predict_proba(test_x)[:, 1]
    fpr, tpr,_ = roc_curve(test_y, predict_prob)
    area_uc = auc(fpr, tpr)
    
    return fpr, tpr, area_uc
    
def classification(data, model_arr, model_name_arr, splits=10):
    stratifiedKFold = StratifiedKFold(n_splits=splits, shuffle=True)
    
    data_x = data.iloc[:, 0:(len(data.columns) - 1)]
    data_y = data["income level"]
    
    accuracy_dt = pd.DataFrame([], columns=model_name_arr)
    conf_mat_dt = pd.DataFrame([], columns=model_name_arr)
    fpr_dt = pd.DataFrame([], columns=model_name_arr)
    tpr_dt = pd.DataFrame([], columns=model_name_arr)
    auc_dt = pd.DataFrame([], columns=model_name_arr)
    
    for train_i, test_i in stratifiedKFold.split(data_x, data_y):
        accuracy_arr = []
        conf_mat_arr = []
        fpr_arr = []
        tpr_arr = []
        auc_arr = []
        
        for i in range(0, len(model_arr)):
            model = model_arr[i]
            model.fit(data_x.iloc[train_i], data_y.iloc[train_i])
        
            predict = model.predict(data_x.iloc[test_i])
        
            accuracy = accuracy_score(predict, data_y.iloc[test_i])
            accuracy_arr.append(accuracy)
            
            mat_conf = confusion_matrix(predict, data_y.iloc[test_i])
            conf_mat_arr.append(mat_conf)
            
            fpr, tpr, auc = do_roc_curve(model, data_x.iloc[test_i], data_y.iloc[test_i])
            fpr_arr.append(fpr)
            tpr_arr.append(tpr)
            auc_arr.append(auc)
        
        acc_dict = {}
        
        for i in range(0, len(model_name_arr)):
            acc_dict.update({model_name_arr[i]: accuracy_arr[i]})
                        
        accuracy_dt = accuracy_dt.append([acc_dict])
        
        conf_mat_dict = {}
        
        for i in range(0, len(model_name_arr)):
            conf_mat_dict.update({model_name_arr[i]: conf_mat_arr[i]})
                        
        conf_mat_dt = conf_mat_dt.append([conf_mat_dict])
        
        fpr_dict = {}
        
        for i in range(0, len(model_name_arr)):
            fpr_dict.update({model_name_arr[i]: fpr_arr[i]})
                        
        fpr_dt = fpr_dt.append([fpr_dict])
        
        tpr_dict = {}
        
        for i in range(0, len(model_name_arr)):
            tpr_dict.update({model_name_arr[i]: tpr_arr[i]})
                        
        tpr_dt = tpr_dt.append([tpr_dict])
        
        auc_dict = {}
        
        for i in range(0, len(model_name_arr)):
            auc_dict.update({model_name_arr[i]: auc_arr[i]})
                        
        auc_dt = auc_dt.append([auc_dict])
        
    return accuracy_dt, conf_mat_dt, fpr_dt, tpr_dt, auc_dt

def fit_model(data, model):
    # Separa os dados em input/output
    data_x = data.iloc[:, 0:(len(data.columns) - 1)]
    data_y = data["income level"]
    
    model.fit(data_x, data_y)
    
    return model

def evaluate_model(data, model):
    # Separa os dados em input/output
    data_x = data.iloc[:, 0:(len(data.columns) - 1)]
    data_y = data["income level"]
    
    model.fit(data_x, data_y)
    
    return model

def compare_models(data, model_name_arr, model_arr):
    # Executa os modelos de classificação
    acc_dt, conf_mat_dt, fpr_dt, tpr_dt, auc_dt = classification(data, model_arr, model_name_arr)
    
    # Imprime os resultados finais
    for i in range(0, len(model_name_arr)):
        print("Model ", model_name_arr[i],
              " | Mean Accuracy: ", np.mean(acc_dt[model_name_arr[i]]), 
              " | Std. Accuracy: ", np.std(acc_dt[model_name_arr[i]]),
              " | AUC: ", np.mean(auc_dt[model_name_arr[i]]))
    
    for i in range(0, len(model_name_arr)):
        print()
        print("#============================== Model " + model_name_arr[i] + " ==============================#")
        print()
        
        draw_conf_matrix(np.mean(conf_mat_dt[model_name_arr[i]]))
    
    color_arr = ["red", "green", "blue", "yellow"]
    draw_roc_curve(model_name_arr, color_arr, fpr_dt, tpr_dt, auc_dt)
    
def predict_model(data, model):
    # Executa o modelo de classificação
    predict = model.predict(data_test)
    
    predict_dt = pd.DataFrame(predict, columns=["Prediction"])
    predict_dt["Prediction"] = predict_dt["Prediction"].astype("str")
    predict_dt["Prediction"] = predict_dt["Prediction"].replace('0.0', '-50000')
    predict_dt["Prediction"] = predict_dt["Prediction"].replace('1.0', '50000+')
    
    return predict_dt

#=========================== Comparando modelos ===============================#

# Carregando os dados
data = load_data("data-07.csv", "attr-07.dat", isTestData=False)
# Pré-processando os dados
data = pre_process_data(data, isTestData=False)
# Balanceando as classes
data = oversampling_data(data)

model_name_arr = ["LR", "KNN", "NB", "TREE"]
model_arr = [LogisticRegression(), KNeighborsClassifier(), GaussianNB(), DecisionTreeClassifier()]
    
# Faz o comparativo entre os modelos
compare_models(data, model_name_arr, model_arr)

#==============================================================================#


#================== Fazendo predição para o melhor modelo =====================#

# Carregando os dados para avaliação
data_test = load_data("data-test.csv", "attr-07.dat", isTestData=True)
# Pré-processando os dados de avaliação
data_test = pre_process_data(data_test, isTestData=True)
# Selecionando os mesmos atributos dos dados em que o modelo será ajustado
data_columns = data.columns[0:(len(data.columns) - 1)]
data_test = data_test[data_columns]

# Cria um modelo do tipo escolhido
best_model = fit_model(data, DecisionTreeClassifier())

# Faz a predição para os dados de avaliação
predict_best_model = predict_model(data_test, best_model)
predict_best_model.to_csv("grupo07.csv", index=False)
print(predict_best_model["Prediction"].value_counts())

#==============================================================================#