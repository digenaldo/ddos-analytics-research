import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Função para calcular o intervalo de confiança
def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean-h, mean+h

# Carregar os dados do CSV
df = pd.read_csv('resultados.csv')

# Extrair os dados necessários
rodadas = df['rodada'].unique()

# Inicializar listas para armazenar resultados
tempos = []
acuracias = []

# Iterar sobre as rodadas para coletar os dados
for rodada in rodadas:
    iteracoes_tempo = eval(df[df['rodada'] == rodada]['iteracoes_tempo'].values[0])
    iteracoes_acuracia = eval(df[df['rodada'] == rodada]['iteracoes_acuracia'].values[0])
    tempos.append(iteracoes_tempo)
    acuracias.append(iteracoes_acuracia)

# Calcular intervalos de confiança para tempos
mean_conf_intervals_tempos = [mean_confidence_interval(tempo) for tempo in tempos]

# Plotar tempos
plt.figure(figsize=(12, 6))
for i, (mean, lower, upper) in enumerate(mean_conf_intervals_tempos):
    plt.plot(range(len(tempos[i])), tempos[i], marker='o', linestyle='', label=f'Rodada {rodadas[i]} Dados')
    plt.hlines(mean, xmin=0, xmax=len(tempos[i])-1, colors='red', label=f'Rodada {rodadas[i]} Média')
    plt.fill_between(range(len(tempos[i])), lower, upper, color='b', alpha=0.1, label=f'Rodada {rodadas[i]} Intervalo de Confiança')

plt.xlabel('Iterações')
plt.ylabel('Tempo (s)')
plt.title('Tempo de Execução com Intervalo de Confiança')
plt.legend(loc='best')  # Ajuste da posição da legenda
plt.tight_layout()
plt.show()

# Calcular intervalos de confiança para acurácia
mean_conf_intervals_acuracias = [mean_confidence_interval(acuracia) for acuracia in acuracias]

# Plotar acurácia
plt.figure(figsize=(12, 6))
for i, (mean, lower, upper) in enumerate(mean_conf_intervals_acuracias):
    plt.plot(range(len(acuracias[i])), acuracias[i], marker='o', linestyle='', label=f'Rodada {rodadas[i]} Dados')
    plt.hlines(mean, xmin=0, xmax=len(acuracias[i])-1, colors='red', label=f'Rodada {rodadas[i]} Média')
    plt.fill_between(range(len(acuracias[i])), lower, upper, color='b', alpha=0.1, label=f'Rodada {rodadas[i]} Intervalo de Confiança')

plt.xlabel('Iterações')
plt.ylabel('Acurácia')
plt.title('Acurácia com Intervalo de Confiança')
plt.legend(loc='best')  # Ajuste da posição da legenda
plt.tight_layout()
plt.show()
