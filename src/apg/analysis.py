import pandas as pd

def analyze_results(results):
    df_results = pd.DataFrame(results)
    print(df_results)
    df_results.to_csv("resultados.csv", index=False)
