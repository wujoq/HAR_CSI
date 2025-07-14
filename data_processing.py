import csiread
import numpy as np
import pandas as pd
import glob
import os

# Folder wyjściowy
output_folder = "csi_data_csv_2"
os.makedirs(output_folder, exist_ok=True)

for dat_file in glob.glob('csi_data/*.dat'):

    csidata = csiread.Intel(dat_file, nrxnum=3, ntxnum=1, pl_size=30)
    csidata.read()

    csi_complex = csidata.get_scaled_csi()
    n_sub, nrx, ntx = csi_complex.shape[1:]

    print(n_sub,nrx, ntx)  # nrx = 3, ntx = 1, n_sub = 90


    rows = []
    for packet_id, csi in enumerate(csi_complex):
        subcarrier_index = 0
        for rx in range(nrx):
            for tx in range(ntx):  # u Ciebie zwykle tx = 0
                for sc in range(n_sub):
                    value = csi[sc,rx,tx]
                    amp = np.abs(value)
                    phase = np.angle(value)
                    rows.append({
                        "packet_id": packet_id,
                        "tx":tx,
                        "rx":rx,
                        "subcarrier": subcarrier_index,
                        "amplitude": amp,
                        "phase": phase
                    })
                    subcarrier_index += 1 
                    if subcarrier_index == 30:
                        subcarrier_index = 0 # unikalne 0–89

    df = pd.DataFrame(rows)

    base_name = os.path.basename(dat_file).replace(".dat", ".csv")
    output_path = os.path.join(output_folder, base_name)
    df.to_csv(output_path, index=False)
    print(f"Zapisano: {output_path}")
