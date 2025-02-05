import pandas as pd
import random
from tabulate import tabulate

# Fungsi untuk menghitung Levenshtein Distance dengan detail operasi
def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    operations = [[[] for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
                operations[i][j] = ['insertion'] * j
            elif j == 0:
                dp[i][j] = i
                operations[i][j] = ['deletion'] * i
            else:
                substitution_cost = 0 if s1[i - 1] == s2[j - 1] else 1
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                substitution = dp[i - 1][j - 1] + substitution_cost

                dp[i][j] = min(insertion, deletion, substitution)

                if dp[i][j] == substitution:
                    operations[i][j] = operations[i - 1][j - 1] + (['substitution'] if substitution_cost == 1 else [])
                elif dp[i][j] == insertion:
                    operations[i][j] = operations[i][j - 1] + ['insertion']
                else:
                    operations[i][j] = operations[i - 1][j] + ['deletion']

    return dp[m][n], operations[m][n]

# Fungsi penilaian otomatis dengan tampilan operasi

def automatic_grading(key_answer, student_answer, max_score=1.0, penalty_per_edit=0.1):
    distance, ops = levenshtein_distance(key_answer.lower(), student_answer.lower())
    final_score = max(max_score - distance * penalty_per_edit, 0)
    return round(final_score, 2), distance, ops

# Fungsi utama untuk menjalankan penilaian soal esai
def main():
    # Membaca file Excel
    file_path = "soal-kunci.xlsx"
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print("File Excel tidak ditemukan. Pastikan file bernama 'soal-kunci.xlsx' ada di direktori yang sama.")
        return

    # Validasi format kolom
    if not {'Soal', 'Kunci Jawaban'}.issubset(df.columns):
        print("File Excel harus memiliki kolom 'Soal' dan 'Kunci Jawaban'.")
        return

    # Pilih 5 soal secara acak
    soal_terpilih = df.sample(5).reset_index(drop=True)

    print("\nSelamat datang di Sistem Penilaian Soal Esai")
    print("\nSilakan jawab pertanyaan berikut:\n")

    total_score = 0
    hasil_penilaian = []

    for i, row in soal_terpilih.iterrows():
        soal = row['Soal']
        kunci_jawaban = row['Kunci Jawaban']

        print(f"{i + 1}. {soal}")
        jawaban_siswa = input("Jawaban Anda: ")

        # Penilaian jawaban siswa
        score, distance, ops = automatic_grading(kunci_jawaban, jawaban_siswa)
        total_score += score
        hasil_penilaian.append([soal, kunci_jawaban, jawaban_siswa, distance, ", ".join(ops), score])

    # Tampilkan hasil penilaian
    print("\nHasil Penilaian:")
    print(tabulate(hasil_penilaian, headers=["Soal", "Kunci Jawaban", "Jawaban Siswa", "Jarak Levenshtein", "Operasi Levenshtein", "Skor"], tablefmt="grid"))
    print(f"\nTotal Skor Anda: {total_score} / 5.0")

if __name__ == "__main__":
    main()
