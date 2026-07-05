"""
automate_Nama-siswa.py

File ini merupakan konversi dari seluruh tahapan preprocessing manual yang telah
dilakukan pada notebook `Eksperimen_Nama-siswa.ipynb`, dibungkus dalam sebuah
fungsi otomatis `preprocess_data()` sehingga dapat dijalankan ulang tanpa
intervensi manual dan mengembalikan data yang siap dilatih.

Cara pakai (CLI):
    python automate_Nama-siswa.py --input ../breast_cancer_raw.csv --output breast_cancer_preprocessing

Cara pakai (import sebagai modul):
    from automate_Nama_siswa import preprocess_data
    X_train, X_test, y_train, y_test = preprocess_data("breast_cancer_raw.csv")
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def cap_outliers_iqr(data: pd.DataFrame, columns) -> pd.DataFrame:
    """Menangani outlier menggunakan metode IQR (capping / winsorizing)."""
    data = data.copy()
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower=lower, upper=upper)
    return data


def preprocess_data(input_path: str, output_dir: str = None, test_size: float = 0.2,
                     random_state: int = 42, save: bool = True):
    """
    Melakukan seluruh tahapan preprocessing secara otomatis:
    1. Load data mentah
    2. Menangani missing value (imputasi median)
    3. Menghapus data duplikat
    4. Menangani outlier (IQR capping)
    5. Split train-test
    6. Standarisasi fitur (StandardScaler)
    7. (Opsional) Menyimpan hasil ke CSV

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
    """
    df = pd.read_csv(input_path)

    df_clean = df.copy()
    num_cols = df_clean.columns.drop("target")

    # 1. Missing value -> imputasi median
    for col in num_cols:
        if df_clean[col].isna().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 2. Hapus duplikat
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # 3. Outlier handling
    df_clean = cap_outliers_iqr(df_clean, num_cols)

    # 4. Split fitur & target
    X = df_clean.drop(columns=["target"])
    y = df_clean["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 5. Standarisasi
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    if save:
        out_dir = output_dir or "breast_cancer_preprocessing"
        os.makedirs(out_dir, exist_ok=True)

        train_final = X_train_scaled.copy()
        train_final["target"] = y_train.values
        test_final = X_test_scaled.copy()
        test_final["target"] = y_test.values

        train_final.to_csv(os.path.join(out_dir, "train.csv"), index=False)
        test_final.to_csv(os.path.join(out_dir, "test.csv"), index=False)
        print(f"[automate_Nama-siswa] Data preprocessing tersimpan di '{out_dir}/'")
        print(f"[automate_Nama-siswa] train: {train_final.shape} | test: {test_final.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test


def main():
    parser = argparse.ArgumentParser(description="Automasi preprocessing dataset breast cancer.")
    parser.add_argument("--input", type=str, default="../breast_cancer_raw.csv",
                         help="Path ke file dataset mentah (CSV)")
    parser.add_argument("--output", type=str, default="breast_cancer_preprocessing",
                         help="Folder output untuk menyimpan hasil preprocessing")
    args = parser.parse_args()

    preprocess_data(input_path=args.input, output_dir=args.output)


if __name__ == "__main__":
    main()
