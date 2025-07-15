import os
import pandas as pd
import numpy as np

# filepath: c:\Programming\WiFi_sensig\plots_2.py

# Wczytaj dane z pliku CSV
filename = 'csi_data_csv_2/standing.csv'  # zmień na inny plik jeśli chcesz
df = pd.read_csv(filename)

# Wyciągnij label z nazwy pliku
label = os.path.splitext(os.path.basename(filename))[0]

# Ustal zakresy pakietów co 500
packet_min = df['packet_id'].min()
packet_max = df['packet_id'].max()
step = 500

# Pobierz unikalne kombinacje tx i rx
tx_rx_pairs = df[['tx', 'rx']].drop_duplicates().values

# Upewnij się, że foldery istnieją
os.makedirs('dataset/x', exist_ok=True)
os.makedirs('dataset/y', exist_ok=True)

counter = 0
for tx, rx in tx_rx_pairs:
    df_txrx = df[(df['tx'] == tx) & (df['rx'] == rx)]
    for start in range(packet_min, packet_max + 1, step):
        end = start + step - 1
        df_slice = df_txrx[(df_txrx['packet_id'] >= start) & (df_txrx['packet_id'] <= end)]
        if df_slice.empty:
            continue
        # Dodaj kolumnę z packet_id % 500
        df_slice = df_slice.copy()
        df_slice['packet_mod'] = df_slice['packet_id'] % 500
        # Pivot: subcarrier x packet_mod (czyli zawsze 0-499)
        heatmap_data = df_slice.pivot_table(index='subcarrier', columns='packet_mod', values='amplitude', aggfunc='mean')
        heatmap_data = heatmap_data.sort_index(axis=0)  # sortuj subcarrier
        heatmap_data = heatmap_data.reindex(columns=range(0, step), fill_value=0)  # zawsze 500 kolumn
        # Zapisz dane i label
        x_path = f'dataset/x/{label}_{tx}_{rx}_{start}_{end}_{counter}.csv'
        y_path = f'dataset/y/{label}_{tx}_{rx}_{start}_{end}_{counter}.csv'
        heatmap_data.to_csv(x_path, index=False)
        pd.DataFrame([label]).to_csv(y_path, index=False, header=False)
        counter += 1