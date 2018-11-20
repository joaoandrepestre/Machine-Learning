#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 18:33:57 2018

@author: joaoandrep
"""

import pandas as pd
import numpy as np

# Importar datasets ------------------------
dataset_train = pd.read_csv('../data/train.csv')
dataset_test = pd.read_csv('../data/test.csv')
#-------------------------------------------

# Remover strings -----------------------------------------------------------------------------------
dataset_train = dataset_train.drop(labels=['tipo', 'bairro', 'tipo_vendedor','diferenciais'],axis=1)
dataset_test = dataset_test.drop(labels=['tipo', 'bairro', 'tipo_vendedor','diferenciais'],axis=1)
#----------------------------------------------------------------------------------------------------

# Remove outliers -----------------------------------------------------------------------------------------------------
dataset_train = dataset_train[np.abs(dataset_train.preco-dataset_train.preco.mean()) <= (3*dataset_train.preco.std())]
#----------------------------------------------------------------------------------------------------------------------

# Separar variáveis dos preços dos imóveis --------
X_train = dataset_train.iloc[:,1:-1].values
y_train = dataset_train.iloc[:,-1].values

X_test = dataset_test.iloc[:,1:].values
#--------------------------------------------------

# Regressão polinomial ------------------------------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Encontra parametros do polinomio para X -----------------
poly_feat = PolynomialFeatures(degree = 2)
X_poly_train = poly_feat.fit_transform(X_train)
X_poly_test = poly_feat.transform(X_test)
#----------------------------------------------------------

# Define regressor com os parametros polinomiais ----------
regressor = LinearRegression()
regressor.fit(X_poly_train, y_train)
#----------------------------------------------------------

# Encontra y previsto pelo regressor --------------------
y_pred_train = regressor.predict(X_poly_train)
y_pred_test = regressor.predict(X_poly_test)
#--------------------------------------------------------

# Análise de erro ------------------------------------------
import math
from sklearn.metrics import mean_squared_error, r2_score

print('\nDesempenho no conjunto de treinamento:')
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y_train, y_pred_train)))
print('R2   = %.3f' %                     r2_score(y_train, y_pred_train) )
#-----------------------------------------------------------

# Constroi arquivo de resposta -----------------------------
d = {'Id': pd.Series(dataset_test.iloc[:,0].values),
  'preco': pd.Series(y_pred_test)}

df = pd.DataFrame(d)

df.to_csv('../resposta.csv',float_format='%.2f',header=['Id','preco'],index=False)
#-----------------------------------------------------------