# Analisis Titik Perubahan (Change Points) Tren Sentimen Media Sosial Menggunakan Spline Truncated

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Math_Core-013243?style=for-the-badge&logo=numpy)
![Transformers](https://img.shields.io/badge/Hugging_Face-IndoBERT-yellow?style=for-the-badge&logo=huggingface)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-239120?style=for-the-badge&logo=python)

## 📌 Latar Belakang
Di era digital, opini publik di media sosial bersifat sangat fluktuatif dan dinamis. Pendekatan statistik konvensional (seperti Regresi Linier) sering kali gagal memodelkan pola data ini karena memaksakan garis lurus global pada data yang perilakunya berubah-ubah.

Proyek ini bertujuan menerapkan metode **Regresi Nonparametrik Spline Truncated Multi-Knot** untuk mendeteksi **Titik-Titik Perubahan (Change Points/Knots)** secara matematis. Titik-titik ini merepresentasikan momen krusial di mana sentimen publik berubah drastis (misal: dari positif menjadi negatif) akibat peristiwa tertentu.

## 🛠️ Metodologi & Pendekatan Numerik

Proyek ini menggabungkan **Natural Language Processing (NLP)** dan **Metode Numerik**:

1.  **Data Acquisition:** Scraping komentar TikTok pada topik viral (Isu "Wapres/Gibran" & Bencana) [Link video yang di Analysis](https://www.tiktok.com/@antaranews/video/7580280925297184007?&t=1765172525354).
2.  **Sentiment Scoring (NLP):**
    * Model sentimen yang digunakan adalah hasil fine-tuning IndoBERT ([`zekiell/indobert-tiktok-political-sentiment`](https://huggingface.co/zekiell/indobert-tiktok-political-sentiment)).
    * Konversi teks ke skalar numerik kontinu (-1 s.d 1) menggunakan rumus probabilitas: $Score = (1 * P_{positive}) + (0 * P_{neutral}) + (-1 * P_{negative})$.
3.  **Numerical Method (The Core):**
    * **Spline Truncated Multi-Knot:** Membangun model regresi potongan (piecewise) yang sangat fleksibel dengan menggunakan lebih dari satu titik potong (knot) untuk menangkap pola data yang kompleks.
    * **Manual Matrix Operation:** Perhitungan koefisien regresi ($\beta$) dilakukan menggunakan operasi matriks OLS (Ordinary Least Square) dengan library `NumPy`, **tanpa** library instan statistik.
    * **Optimasi Knot:** Menggunakan metode **Generalized Cross Validation (GCV)** untuk mencari lokasi dan jumlah titik knot optimal dengan error terkecil.
    * **Visualisasi Data:** Menggunakan library `Matplotlib` untuk memetakan tren sentimen asli vs prediksi model spline serta menonjolkan lokasi change points.

## 🧮 Rumus Utama

Model Spline Truncated Linear dengan titik knot jamak ($K_1, K_2, ..., K_m$) diformulasikan sebagai:

$$y = \beta_0 + \beta_1x + \sum_{k=1}^{m} \beta_{k+1}(x-K_k)_+ $$

Dimana fungsi truncated untuk setiap knot ke-$k$ didefinisikan sebagai:

$$
(x - K_k)_+ = 
\begin{cases} 
(x - K_k) & \text{jika } x \ge K_k \\
0 & \text{jika } x < K_k 
\end{cases}
$$
