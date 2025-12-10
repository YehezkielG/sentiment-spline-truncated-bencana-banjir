# Analisis Titik Perubahan (Change Points) Tren Sentimen Media Sosial Menggunakan Spline Truncated 

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Math_Core-013243?style=for-the-badge&logo=numpy)
![Transformers](https://img.shields.io/badge/Hugging_Face-IndoBERT-yellow?style=for-the-badge&logo=huggingface)
![Status](https://img.shields.io/badge/Status-In_Progress-orange?style=for-the-badge)

> **Proyek Rekayasa Ide - Mata Kuliah Metode Numerik** > Universitas Negeri Medan (UNIMED)

## 📌 Latar Belakang
Di era digital, opini publik di media sosial bersifat sangat fluktuatif dan dinamis. Pendekatan statistik konvensional (seperti Regresi Linier) sering kali gagal memodelkan pola data ini karena memaksakan garis lurus global pada data yang perilakunya berubah-ubah.

Proyek ini bertujuan menerapkan metode **Regresi Nonparametrik Spline Truncated** untuk mendeteksi **Titik Perubahan (Change Points/Knots)** secara matematis. Titik ini merepresentasikan momen krusial di mana sentimen publik berubah drastis (misal: dari positif menjadi negatif) akibat peristiwa tertentu.


## 🛠️ Metodologi & Pendekatan Numerik

Proyek ini menggabungkan **Natural Language Processing (NLP)** dan **Metode Numerik**:

1.  **Data Acquisition:** Scraping komentar TikTok pada topik viral (Isu "Wapres/Gibran" & Bencana) [Link video yang di Analysis](https://www.tiktok.com/@antaranews/video/7580280925297184007?&t=1765172525354).
2.  **Sentiment Scoring (NLP):**
    * Menggunakan model Pre-trained **IndoBERT** (`w11wo/indonesian-roberta-base-sentiment-classifier`).
    * Konversi teks ke skalar numerik kontinu (-1 s.d 1) menggunakan rumus probabilitas: $Score = (1 * P_{positive}) + (0 * P_{neutral}) + (-1 * P_{negative})$.
3.  **Numerical Method (The Core):**
    * **Spline Truncated:** Membangun model regresi potongan (piecewise) yang fleksibel.
    * **Manual Matrix Operation:** Perhitungan koefisien regresi ($\beta$) dilakukan menggunakan operasi matriks OLS (Ordinary Least Square) dengan library `NumPy`, **tanpa** library instan statistik.
    * **Optimasi Knot:** Menggunakan metode **Generalized Cross Validation (GCV)** untuk mencari titik knot optimal dengan error terkecil.

## 🧮 Rumus Utama

Model Spline Truncated Linear dengan satu titik knot ($K$) diformulasikan sebagai:

$$y = \beta_0 + \beta_1x + \beta_2(x-K)_+ + \epsilon$$

Dimana fungsi truncated didefinisikan sebagai:

$$
(x - K)_+ = 
\begin{cases} 
(x - K) & \text{jika } x \ge K \\
0 & \text{jika } x < K 
\end{cases}
$$
