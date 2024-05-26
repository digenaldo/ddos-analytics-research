import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
mean_tempos = []
ci_lower_tempos = []
ci_upper_tempos = []
mean_acuracias = []
ci_lower_acuracias = []
ci_upper_acuracias = []

# Iterar sobre as rodadas para coletar os dados
for rodada in rodadas:
    iteracoes_tempo = eval(df[df['rodada'] == rodada]['iteracoes_tempo'].values[0])
    iteracoes_acuracia = eval(df[df['rodada'] == rodada]['iteracoes_acuracia'].values[0])
    tempos.append(iteracoes_tempo)
    acuracias.append(iteracoes_acuracia)
    
    mean, lower, upper = mean_confidence_interval(iteracoes_tempo)
    mean_tempos.append(mean)
    ci_lower_tempos.append(lower)
    ci_upper_tempos.append(upper)
    
    mean, lower, upper = mean_confidence_interval(iteracoes_acuracia)
    mean_acuracias.append(mean)
    ci_lower_acuracias.append(lower)
    ci_upper_acuracias.append(upper)

# Plotar gráfico de barras com intervalos de confiança para tempos
plt.figure(figsize=(12, 6))
bars = plt.bar(rodadas, mean_tempos, yerr=[np.array(mean_tempos)-np.array(ci_lower_tempos), np.array(ci_upper_tempos)-np.array(mean_tempos)], capsize=5, alpha=0.7, color='skyblue', ecolor='black', error_kw=dict(lw=1.5, capsize=5))

# Adicionar anotações aos intervalos de confiança dos tempos
for bar, mean, lower, upper in zip(bars, mean_tempos, ci_lower_tempos, ci_upper_tempos):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, upper, f'{upper:.2f}', ha='center', va='bottom', color='black', fontsize=10)
    plt.text(bar.get_x() + bar.get_width() / 2, lower, f'{lower:.2f}', ha='center', va='top', color='black', fontsize=10)

plt.xlabel('Rodadas')
plt.ylabel('Tempo (s)')
plt.title('Tempo de Execução com Intervalos de Confiança')
plt.tight_layout()
plt.show()

# Plotar gráfico de barras com intervalos de confiança para acurácias
plt.figure(figsize=(12, 6))
bars = plt.bar(rodadas, mean_acuracias, yerr=[np.array(mean_acuracias)-np.array(ci_lower_acuracias), np.array(ci_upper_acuracias)-np.array(mean_acuracias)], capsize=5, alpha=0.7, color='lightgreen', ecolor='black', error_kw=dict(lw=1.5, capsize=5))

# Adicionar anotações aos intervalos de confiança das acurácias
for bar, mean, lower, upper in zip(bars, mean_acuracias, ci_lower_acuracias, ci_upper_acuracias):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, upper, f'{upper:.7f}', ha='center', va='bottom', color='black', fontsize=10)
    plt.text(bar.get_x() + bar.get_width() / 2, lower, f'{lower:.7f}', ha='center', va='top', color='black', fontsize=10)

plt.xlabel('Rodadas')
plt.ylabel('Acurácia')
plt.title('Acurácia com Intervalos de Confiança')
plt.tight_layout()
plt.show()

# Preparar dados para gráfico de violino para tempos
data = []
for i, rodada in enumerate(rodadas):
    for tempo in tempos[i]:
        data.append([rodada, tempo])

df_violin = pd.DataFrame(data, columns=['Rodada', 'Tempo'])

# Plotar gráfico de violino para tempos
plt.figure(figsize=(12, 6))
sns.violinplot(x='Rodada', y='Tempo', data=df_violin, inner='point', palette='muted')
plt.xlabel('Rodadas')
plt.ylabel('Tempo (s)')
plt.title('Distribuição do Tempo de Execução por Rodada')
plt.tight_layout()
plt.show()

# Preparar dados para gráfico de violino para acurácias
data = []
for i, rodada in enumerate(rodadas):
    for acuracia in acuracias[i]:
        data.append([rodada, acuracia])

df_violin_acuracia = pd.DataFrame(data, columns=['Rodada', 'Acurácia'])

# Plotar gráfico de violino para acurácias
plt.figure(figsize=(12, 6))
sns.violinplot(x='Rodada', y='Acurácia', data=df_violin_acuracia, inner='point', palette='muted')
plt.xlabel('Rodadas')
plt.ylabel('Acurácia')
plt.title('Distribuição da Acurácia por Rodada')
plt.tight_layout()
plt.show()