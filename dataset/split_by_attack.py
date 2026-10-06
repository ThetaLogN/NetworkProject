import os
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_dir, 'CICIDS2017_sample.csv')
    output_base = os.path.join(base_dir, 'by_attack')
    single_dir = os.path.join(output_base, 'single_classes')
    binary_dir = os.path.join(output_base, 'binary_with_benign')

    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(binary_dir, exist_ok=True)

    print(f"Caricamento di {input_csv}...")
    df = pd.read_csv(input_csv)

    print("Distribuzione classi originale:")
    counts = df['Label'].value_counts()
    print(counts)

    benign_df = df[df['Label'] == 'BENIGN']

    # 1. Salva ogni classe singolarmente
    for label in counts.index:
        label_df = df[df['Label'] == label]
        safe_name = label.replace('/', '_').replace(' ', '_')
        single_path = os.path.join(single_dir, f"{safe_name}.csv")
        label_df.to_csv(single_path, index=False)
        print(f"[Singola classe] Salvato: {single_path} ({len(label_df):,} righe)")

    # 2. Salva dataset binari (Attacco + BENIGN) per training mirato
    for label in counts.index:
        if label == 'BENIGN':
            continue
        attack_df = df[df['Label'] == label]
        binary_df = pd.concat([benign_df, attack_df], ignore_index=True)
        safe_name = label.replace('/', '_').replace(' ', '_')
        binary_path = os.path.join(binary_dir, f"{safe_name}_vs_BENIGN.csv")
        binary_df.to_csv(binary_path, index=False)
        print(f"[Dataset binario] Salvato: {binary_path} ({len(binary_df):,} righe: {len(benign_df):,} BENIGN + {len(attack_df):,} {label})")

    print("\nOperazione completata con successo!")

if __name__ == '__main__':
    main()
