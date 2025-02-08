from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import random
import re
import string
from sentence_transformers import SentenceTransformer, util
import torch
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-123'  # Needed for session

# Initialize the semantic model at startup
try:
    semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    print(f"Warning: Could not load semantic model: {str(e)}")
    semantic_model = None

# Fungsi untuk membersihkan teks dari tanda baca dan mengubah ke lowercase
def clean_text(text):
    # Hapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Ubah ke lowercase dan hapus spasi berlebih
    text = ' '.join(text.lower().split())
    return text

# Fungsi untuk menghitung Levenshtein Distance dengan detail operasi
def levenshtein_distance(s1, s2):
    """
    Menghitung Levenshtein Distance antara dua string dan mencatat operasi yang dilakukan.
    
    Levenshtein Distance adalah jumlah minimum operasi yang diperlukan untuk mengubah satu string 
    menjadi string lain. Operasi yang diperbolehkan adalah:
    1. Insertion (Penyisipan): Menambah karakter
    2. Deletion (Penghapusan): Menghapus karakter
    3. Substitution (Penggantian): Mengganti satu karakter dengan karakter lain
    
    Algoritma menggunakan dynamic programming dengan matriks berukuran (m+1) x (n+1)
    dimana m dan n adalah panjang dari kedua string.
    
    Args:
        s1 (str): String pertama (string sumber)
        s2 (str): String kedua (string target)
    
    Returns:
        tuple: (jarak_levenshtein, daftar_operasi)
            - jarak_levenshtein: Jumlah minimum operasi yang diperlukan
            - daftar_operasi: List berisi operasi yang dilakukan (insertion/deletion/substitution)
    """
    # Inisialisasi matriks dengan ukuran (m+1) x (n+1)
    # m = panjang string pertama, n = panjang string kedua
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Matriks untuk menyimpan operasi yang dilakukan pada setiap sel
    operations = [[[] for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Inisialisasi kasus dasar:
    # 1. Mengubah string kosong menjadi s2[0...j]
    for j in range(n + 1):
        dp[0][j] = j  # Membutuhkan j operasi insertion
        operations[0][j] = ['insertion'] * j
    
    # 2. Mengubah s1[0...i] menjadi string kosong
    for i in range(m + 1):
        dp[i][0] = i  # Membutuhkan i operasi deletion
        operations[i][0] = ['deletion'] * i
    
    # Mengisi matriks dengan dynamic programming
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Jika karakter sama, tidak ada biaya (substitution_cost = 0)
            # Jika berbeda, ada biaya 1 untuk substitusi
            substitution_cost = 0 if s1[i - 1] == s2[j - 1] else 1
            
            # Hitung biaya untuk setiap operasi yang mungkin:
            insertion = dp[i][j - 1] + 1      # Menyisipkan karakter dari s2
            deletion = dp[i - 1][j] + 1       # Menghapus karakter dari s1
            substitution = dp[i - 1][j - 1] + substitution_cost  # Mengganti karakter
            
            # Ambil operasi dengan biaya minimum
            dp[i][j] = min(insertion, deletion, substitution)
            
            # Catat operasi yang dilakukan berdasarkan nilai minimum
            if dp[i][j] == substitution:
                # Salin operasi sebelumnya dan tambahkan substitusi jika karakternya berbeda
                operations[i][j] = operations[i - 1][j - 1] + (['substitution'] if substitution_cost == 1 else [])
            elif dp[i][j] == insertion:
                # Salin operasi sebelumnya dan tambahkan insertion
                operations[i][j] = operations[i][j - 1] + ['insertion']
            else:  # dp[i][j] == deletion
                # Salin operasi sebelumnya dan tambahkan deletion
                operations[i][j] = operations[i - 1][j] + ['deletion']
    
    # Nilai di dp[m][n] adalah Levenshtein Distance final
    # operations[m][n] berisi urutan operasi yang dilakukan
    return dp[m][n], operations[m][n]

# Fungsi untuk menghitung Jaro Similarity
def jaro_similarity(s1, s2):
    """
    Menghitung Jaro Similarity antara dua string.
    
    Jaro Similarity mengukur kesamaan antara dua string berdasarkan:
    1. Jumlah karakter yang sama
    2. Jumlah transposisi
    3. Panjang kedua string
    
    Args:
        s1 (str): String pertama
        s2 (str): String kedua
    
    Returns:
        float: Nilai similarity antara 0 dan 1
    """
    # Jika kedua string kosong, return 1
    if not s1 and not s2:
        return 1.0
    
    # Jika salah satu string kosong, return 0
    if not s1 or not s2:
        return 0.0
    
    # Hitung jarak maksimum untuk matching
    match_distance = (max(len(s1), len(s2)) // 2) - 1
    
    # Inisialisasi array untuk tracking matches
    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    
    # Hitung matches
    matches = 0
    for i in range(len(s1)):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(s2))
        
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
    
    if matches == 0:
        return 0.0
    
    # Hitung transposisi
    transpositions = 0
    j = 0
    for i in range(len(s1)):
        if s1_matches[i]:
            while j < len(s2) and not s2_matches[j]:
                j += 1
            if j < len(s2) and s1[i] != s2[j]:
                transpositions += 1
            j += 1
    
    transpositions = transpositions // 2
    
    # Hitung Jaro Similarity
    return (matches / len(s1) + matches / len(s2) + (matches - transpositions) / matches) / 3.0

def jaro_winkler_similarity(s1, s2, p=0.1):
    """
    Menghitung Jaro-Winkler Similarity antara dua string.
    
    Jaro-Winkler memodifikasi Jaro Similarity dengan memberikan bobot lebih
    untuk kecocokan di awal string.
    
    Args:
        s1 (str): String pertama
        s2 (str): String kedua
        p (float): Scaling factor untuk prefix matches (default 0.1)
    
    Returns:
        float: Nilai similarity antara 0 dan 1
    """
    # Hitung Jaro Similarity
    jaro_sim = jaro_similarity(s1, s2)
    
    # Hitung panjang prefix yang sama (maksimal 4 karakter)
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    
    # Hitung Jaro-Winkler Similarity
    return jaro_sim + (prefix_len * p * (1 - jaro_sim))

def get_jaro_winkler_details(s1, s2):
    """
    Menghitung detail operasi Jaro-Winkler antara dua string.
    """
    # Jika salah satu string kosong
    if not s1 or not s2:
        return {
            'matching_chars': [],
            'transpositions': [],
            'prefix_length': 0,
            'operations': ['String kosong']
        }
    
    # Hitung jarak maksimum untuk matching
    match_distance = (max(len(s1), len(s2)) // 2) - 1
    match_distance = max(0, match_distance)
    
    # Inisialisasi array untuk tracking matches
    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    
    # Temukan karakter yang cocok
    matching_chars = []
    for i in range(len(s1)):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(s2))
        
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matching_chars.append((i, j, s1[i]))
                break
    
    # Temukan transposisi
    transpositions = []
    j = 0
    for i in range(len(s1)):
        if s1_matches[i]:
            while j < len(s2) and not s2_matches[j]:
                j += 1
            if j < len(s2) and s1[i] != s2[j]:
                transpositions.append((i, j))
            j += 1
    
    # Hitung panjang prefix yang sama
    prefix_length = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_length += 1
        else:
            break
    
    # Buat daftar operasi
    operations = []
    if matching_chars:
        operations.append(f"Menemukan {len(matching_chars)} karakter yang cocok")
    if transpositions:
        operations.append(f"Terdapat {len(transpositions)} transposisi")
    if prefix_length > 0:
        operations.append(f"Prefix yang sama: {prefix_length} karakter")
    
    return {
        'matching_chars': matching_chars,
        'transpositions': transpositions,
        'prefix_length': prefix_length,
        'operations': operations
    }

def calculate_semantic_similarity(s1, s2):
    """
    Menghitung semantic similarity antara dua teks menggunakan model BERT.
    
    Args:
        s1 (str): Teks pertama
        s2 (str): Teks kedua
    
    Returns:
        float: Nilai similarity antara 0 dan 1
    """
    if semantic_model is None:
        print("Warning: Semantic model not available, falling back to text similarity")
        return jaro_winkler_similarity(s1, s2)  # Fallback to Jaro-Winkler
        
    try:
        # Encode kedua teks
        embedding1 = semantic_model.encode(s1, convert_to_tensor=True)
        embedding2 = semantic_model.encode(s2, convert_to_tensor=True)
        
        # Hitung cosine similarity
        similarity = util.pytorch_cos_sim(embedding1, embedding2)
        
        return float(similarity[0][0])  # Convert tensor to float
    except Exception as e:
        print(f"Error in semantic similarity calculation: {str(e)}")
        return jaro_winkler_similarity(s1, s2)  # Fallback to Jaro-Winkler

def automatic_grading_combined(key_answer, student_answer, max_score=1.0):
    """
    Menghitung skor menggunakan kombinasi Levenshtein, Jaro-Winkler, dan Semantic Similarity
    """
    # Bersihkan teks
    clean_key = clean_text(key_answer)
    clean_student = clean_text(student_answer)
    
    # Hitung Levenshtein distance dan normalisasi
    lev_distance, _ = levenshtein_distance(clean_key, clean_student)
    max_len = max(len(clean_key), len(clean_student))
    lev_similarity = 1 - (lev_distance / max_len if max_len > 0 else 0)
    
    # Hitung Jaro-Winkler similarity
    jw_similarity = jaro_winkler_similarity(clean_key, clean_student)
    
    # Hitung semantic similarity
    semantic_sim = calculate_semantic_similarity(key_answer, student_answer)
    
    # Kombinasikan semua metrik dengan bobot
    # Berikan bobot lebih tinggi untuk semantic similarity
    combined_score = (0.25 * lev_similarity + 
                     0.25 * jw_similarity + 
                     0.5 * semantic_sim)  # Semantic similarity memiliki bobot 50%
    
    # Normalisasi skor akhir
    final_score = combined_score * max_score
    
    return {
        'score': round(final_score, 2),
        'levenshtein_similarity': round(lev_similarity, 3),
        'jaro_winkler_similarity': round(jw_similarity, 3),
        'semantic_similarity': round(semantic_sim, 3),
        'explanation': {
            'levenshtein': f"Levenshtein Similarity: {round(lev_similarity * 100, 1)}%",
            'jaro_winkler': f"Jaro-Winkler Similarity: {round(jw_similarity * 100, 1)}%",
            'semantic': f"Semantic Similarity: {round(semantic_sim * 100, 1)}%",
            'final': f"Final Score: {round(final_score, 2)} / {max_score}"
        }
    }

# Fungsi penilaian otomatis dengan tampilan operasi
def automatic_grading(key_answer, student_answer, max_score=1.0):
    # Bersihkan teks dari tanda baca dan ubah ke lowercase
    clean_key = clean_text(key_answer)
    clean_student = clean_text(student_answer)
    
    # Split jawaban menjadi kata-kata
    key_words = set(clean_key.split())
    student_words = set(clean_student.split())
    
    # Hitung kata yang cocok
    matching_words = key_words.intersection(student_words)
    
    # Hitung skor berdasarkan jumlah kata yang cocok
    if matching_words:
        # Minimal 0.3 poin jika ada kata yang cocok
        base_score = 0.3
        # Tambahan skor berdasarkan proporsi kata yang cocok
        word_match_score = len(matching_words) / len(key_words) * 0.7
        final_score = min(base_score + word_match_score, max_score)
    else:
        # Jika tidak ada kata yang cocok, gunakan Levenshtein distance
        distance, ops = levenshtein_distance(clean_key, clean_student)
        max_possible_distance = max(len(clean_key), len(clean_student))
        if max_possible_distance == 0:
            final_score = 0
        else:
            similarity = 1 - (distance / max_possible_distance)
            final_score = similarity * max_score
    
    # Dapatkan distance dan operations untuk ditampilkan
    distance, ops = levenshtein_distance(clean_key, clean_student)
    
    # Tampilkan kata-kata yang cocok dalam hasil
    matching_words_list = list(matching_words)
    
    return round(final_score, 2), distance, ops, matching_words_list

# Route untuk halaman utama
@app.route('/')
def index():
    try:
        # Bersihkan session saat memulai baru
        session.clear()
        
        # Membaca file Excel
        df = pd.read_excel('soal-kunci-clean.xlsx')
        
        # Validasi format kolom
        if not {'Soal', 'Kunci Jawaban'}.issubset(df.columns):
            return "File Excel harus memiliki kolom 'Soal' dan 'Kunci Jawaban'."
        
        # Pilih 5 soal secara acak
        soal_terpilih = df.sample(5).to_dict('records')
        
        # Simpan soal terpilih di session
        session['soal_terpilih'] = soal_terpilih
        
        return render_template('index.html', soal=soal_terpilih)
    
    except FileNotFoundError:
        return "File Excel tidak ditemukan. Pastikan file bernama 'soal-kunci.xlsx' ada di direktori yang sama."
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}"

# Route untuk memproses jawaban
@app.route('/submit', methods=['POST'])
def submit_answers():
    try:
        # Ambil soal dari session
        soal_terpilih = session.get('soal_terpilih', [])
        if not soal_terpilih:
            return redirect(url_for('index'))
        
        hasil_penilaian = []
        total_levenshtein_score = 0
        total_jaro_winkler_score = 0
        
        # Proses setiap jawaban
        for i, soal in enumerate(soal_terpilih):
            jawaban_siswa = request.form.get(f'answer{i}', '')
            kunci_jawaban = soal['Kunci Jawaban']
            
            # Hitung skor dengan kedua metode
            scores = automatic_grading_combined(kunci_jawaban, jawaban_siswa)
            
            total_levenshtein_score += scores['levenshtein_similarity']
            total_jaro_winkler_score += scores['jaro_winkler_similarity']
            
            # Simpan hasil
            hasil_penilaian.append({
                'soal': soal['Soal'],
                'kunci_jawaban': kunci_jawaban,
                'jawaban_siswa': jawaban_siswa,
                'levenshtein_similarity': scores['levenshtein_similarity'],
                'jaro_winkler_similarity': scores['jaro_winkler_similarity'],
                'semantic_similarity': scores['semantic_similarity'],
                'explanation': scores['explanation']
            })
        
        return render_template('results.html', 
                             hasil_penilaian=hasil_penilaian,
                             total_levenshtein_score=round(total_levenshtein_score, 2),
                             total_jaro_winkler_score=round(total_jaro_winkler_score, 2),
                             max_score=5.0)
    
    except Exception as e:
        return f"Terjadi kesalahan saat memproses jawaban: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=8080)
