import yfinance as yf
import math
import matplotlib.pyplot as plt
import numpy as np
from time import time
import opcoes as opc
import pandas as pd
import scipy.stats
from scipy.stats import norm
from datetime import date

def consulta_bc(codigo_bcb):
    '''
    Importa os dados Banco Central através de uma API
    '''
    url = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json'.format(codigo_bcb)
    df = pd.read_json(url)
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    df.set_index('data', inplace=True)
    return df


def calcula_retornos(data: pd.Series, column="Close"):
    '''
    Calcula o retorno para cada intervalo de tempo da série
    '''
    df = data[column].pct_change()
    df.drop(df.head(1).index,inplace=True)
    
    return pd.DataFrame(df)


def cornish (r: pd.Series, z):
    '''
    Função para adptar o z com a ampliação de cornish-fisher
    '''
    s = scipy.stats.skew(r)
    k = scipy.stats.kurtosis(r)
    z = (z +
            (z**2 - 1)*s/6 +
            (z**3 -3*z)*(k-3)/24 -
            (2*z**3 - 5*z)*(s**2)/36
        )
    return z


def is_normal(r, level=0.01):
    """
    Utiliza o Jarque-Bera test para determinar se a distribuição é normal ou não
    O nível de confiança é de 1%, podendo mudar seu parâmetro
    Retorna verdadeiro se a hipotese de normalidade é verdadeira
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def mc_normal(data, S0 = -1, K = -1, r = -1, M = -1, I = -1, retornar = -1):
    '''
    Retorna um DataFrame com as simulações de Monte Carlo
    Ou mesmo o valor de uma opção europeia
    
    Adaptado de Pricing a European Call Option Using Monte Carlo Simulation por Nícolas Mikio
    https://www.datasciencecentral.com/profiles/blogs/pricing-options-using-monte-carlo-simulations
    https://sphelps.net/teaching/fpmf/slides/all.html
    '''
    if S0 == -1:
        S0 = float(input("Qual o preço da ação? "))
        
    if K == -1:
        K = float(input("Qual o preço de exercício? "))
        
    if M == -1:
        M = int(input("Para quantos períodos de tempo? "))
        
    if I == -1:
        I = int(input("Quantas simulações? "))
        
    if r == -1:  
        cdi = consulta_bc(12)
        cdi_recente = cdi.iloc[-1]
        r = (1 + cdi_recente/100).prod()**(252)-1
        #Transformação em taxa composta continuamente
        r = math.log(r+1)
    
    T = M/252

    sigma = data.std()*(252**0.5)
    sigma = sigma[0]

    S = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * (T/M) + sigma * math.sqrt(T/M) * 
                                  np.random.standard_normal((M + 1, I)), axis=0))
    
    if retornar == -1:
        retornar = int(input("DataFrame = 1, call = 2 ou put = 3? "))
    
    if retornar == 1:
        return  pd.DataFrame(S)
    
    if retornar == 2:
        return math.exp(-r * T) * sum(np.maximum(S[-1] - K, 0)) / I
    
    if retornar == 3:
        return math.exp(-r * T) * sum(np.maximum(K - S[-1], 0)) / I


def euro_vanilla_call(data, S=-1, K=-1, T=-1, r=-1):
    '''
    Precifica uma opção de call europeia, com base nos argumentos fornecidos:
    (data, S= spot price, K= strike price, T= time to maturity, r= interest rate).
    
    Os argumentos podem ser digitados no input, ou mesmo incluídos dentro da função.
    
    É utilizado a ampliação de cornish-fisher para corrigir o valor z atribuído pelas argumentos d1 e d2
    
    A taxa livre de risco dada como default pela função é a o CDI composta continuamente,
    porém pode ser modificada na função ao escrever:
    euro_vanilla_call(data, r= valor_da_taxa)
    '''
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    if S == -1:
        S = float(input("Qual o preço da ação? "))
        
    if K == -1:
        K = float(input("Qual o preço de exercício? "))
        
    if T == -1:
        T = int(input("Quantos períodos até a data de expiração? "))
        T = T/252
        
    if r == -1:  
        cdi = consulta_bc(12)
        cdi_recente = cdi.iloc[-1]
        r = (1 + cdi_recente/100).prod()**(252)-1
        #Transformação em taxa composta continuamente
        r = math.log(r+1)
    
    #volatilidade (desvio padrao) anualizado
    sigma = data.std()*(252**0.5)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    d1 = cornish(data, d1)
    d2 = cornish(data, d2)
    
    call = (S * scipy.stats.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * scipy.stats.norm.cdf(d2, 0.0, 1.0))
    
    return call[0].round(2)


def euro_vanilla_put(data, S=-1, K=-1, T=-1, r=-1):
    
    '''
    Precifica uma opção de put europeia, com base nos argumentos fornecidos:
    (data, S= spot price, K= strike price, T= time to maturity, r= interest rate).
    
    Os argumentos podem ser digitados no input, ou mesmo incluídos dentro da função.
    
    É utilizado a ampliação de cornish-fisher para corrigir o valor z atribuído pelas argumentos d1 e d2
    
    A taxa livre de risco dada como default pela função é a o CDI acumulado anual dos anos 2020 a 2021,
    porém pode ser modificada na função ao escrever:
    euro_vanilla_call(data, r= valor_da_taxa)
    '''
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    if S == -1:
        S = float(input("Qual o preço da ação? "))
        
    if K == -1:
        K = float(input("Qual o preço de exercício? "))
        
    if T == -1:
        T = int(input("Quantos períodos até a data de expiração? "))
        T = T/252
        
    if r == -1:  
        cdi = consulta_bc(12)
        cdi_recente = cdi.iloc[-1]
        r = (1 + cdi_recente/100).prod()**(252)-1
        #Transformação em taxa composta continuamente
        r = math.log(r+1)
    
    #volatilidade (desvio padrao) anualizado
    sigma = data.std()*(252**0.5)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    d1 = cornish(data, d1)
    d2 = cornish(data, d2)
    
    put = (K * np.exp(-r * T) * scipy.stats.norm.cdf(-d2, 0.0, 1.0) - S * scipy.stats.norm.cdf(-d1, 0.0, 1.0))
    
    return put[0].round(2)


def exotica(data, S0 = -1, K = -1, r = -1, M = -1, I = -1, retornar = -1):
    '''
    Retorna um DataFrame com as simulações de Monte Carlo, ou mesmo o valor de uma opção exótica
    em que o preço final é a média dos últimos 5 dias úteis    
    '''
    if S0 == -1:
        S0 = float(input("Qual o preço da ação? "))
        
    if K == -1:
        K = float(input("Qual o preço de exercício? "))
        
    if M == -1:
        M = int(input("Para quantos períodos de tempo? "))
        
    if I == -1:
        I = int(input("Quantas simulações? "))
        
    if r == -1:  
        cdi = consulta_bc(12)
        cdi_recente = cdi.iloc[-1]
        r = (1 + cdi_recente/100).prod()**(252)-1
        #Transformação em taxa composta continuamente
        r = math.log(r+1)
        
    T = M/252
        
    sigma = data.std()*(252**0.5)
    sigma = sigma[0]
    


    S = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * (T/M) + sigma * math.sqrt(T/M) * 
                                  np.random.standard_normal((M + 1, I)), axis=0))
    
    if retornar == -1:
        retornar = int(input("DataFrame = 1, call = 2 ou put = 3? "))
    
    S = pd.DataFrame(S)
    
    if retornar == 1:
        return S
    
    if retornar == 2:
        return math.exp(-r * T) * sum(np.maximum(S.iloc[-6:-1].mean() - K, 0)) / I
    
    if retornar == 3:
        return math.exp(-r * T) * sum(np.maximum(K - S.iloc[-6:-1].mean(), 0)) / I
    
    
    
def binaria(data, S0 = -1, K = -1, recompensa =-1, r = -1, M = -1, I = -1, retornar = -1):
    '''
    Retorna o preço de uma opção exótica binária    
    '''
    if S0 == -1:
        S0 = float(input("Qual o preço da ação? "))
        
    if K == -1:
        K = float(input("Qual o preço de exercício? "))
    
    if recompensa == -1:
        recompensa = float(input("Qual o valor da recompensa? "))
        
    if M == -1:
        M = int(input("Para quantos períodos de tempo? "))
        
    if I == -1:
        I = int(input("Quantas simulações? "))
        
    if r == -1:  
        cdi = consulta_bc(12)
        cdi_recente = cdi.iloc[-1]
        r = (1 + cdi_recente/100).prod()**(252)-1
        #Transformação em taxa composta continuamente
        r = math.log(r+1)
       
    T = M/252
    
    sigma = data.std()*(252**0.5)
    sigma = sigma[0]
    

    S = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * (T/M) + sigma * math.sqrt(T/M) * 
                                  np.random.standard_normal((M + 1, I)), axis=0))
    
    if retornar == -1:
        retornar = int(input("higher = 1 ou lower = 2? "))

    #Higher
    if retornar == 1:
        payoffs = np.sum(S[-1] > K)

    #Lower
    if retornar == 2:
        payoffs = np.sum(S[-1] < K)

    option_price = math.exp(-r * T) * (payoffs / I) * recompensa
    
    return option_price