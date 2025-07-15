import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Wczytaj dane z pliku CSV
filename = 'csi_data_csv_2/standing.csv'  # zmień na inny plik jeśli chcesz
df = pd.read_csv(filename)

# Ustal zakresy pakietów co 500
packet_min = df['packet_id'].min()
packet_max = df['packet_id'].max()
step = 500

# Pobierz unikalne kombinacje tx i rx
tx_rx_pairs = df[['tx', 'rx']].drop_duplicates().values

for tx, rx in tx_rx_pairs:
    df_txrx = df[(df['tx'] == tx) & (df['rx'] == rx)]
    for start in range(packet_min, packet_max + 1, step):
        end = start + step - 1
        df_slice = df_txrx[(df_txrx['packet_id'] >= start) & (df_txrx['packet_id'] <= end)]
        if df_slice.empty:
            continue
        heatmap_data = df_slice.pivot_table(index='subcarrier', columns='packet_id', values='amplitude', aggfunc='mean')
        heatmap_data = heatmap_data.sort_index()
        plt.figure(figsize=(12, 6))
        plt.imshow(heatmap_data, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Amplituda')
        plt.xlabel('Packet ID')
        plt.ylabel('Subcarrier')
        plt.title(f'Heatmapa amplitudy: {filename}\nPakiety {start}-{end}, tx={tx}, rx={rx}')
       # plt.yticks(np.arange(0, 91, 10))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()