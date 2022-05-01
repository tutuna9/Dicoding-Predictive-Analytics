# Laporan Proyek Rekomendasi Anime - M. Ichsan Nur Iman

## Project Overview

Sekarang anime mulai disukai oleh banyak orang. Hal ini terjadi bersamaan dengan naiknya kecepatan anime dirilis sehingga untuk orang yang baru mulai menonton anime mungkin akan bingung untuk memilih anime apa yang selanjutnya ingin dia tonton.

Oleh karena itu, saya akan mencoba membuat sistem yang bisa merekomendasikan seseorang berdasarkan genre anime, atau berdasarkan rating yang mereka berikan ke anime yang pernah mereka tonton sebelumnya.

## Business Understanding

### Problem Statement

- Bagaimana cara membuat sistem rekomendasi berdasarkan genre animenya?
- Bagaimana cara membuat sistem rekomendasi berdasarkan rating yang diberi user sebelumnya?

### Goals

- Mengetahui cara membuat sistem rekomendasi berdasarkan genre animenya
- Mengetahui cara membuat sistem rekomendasi berdasarkan rating yang diberi user sebelumnya

### Solution Approach

- Menggunakan Content Based Filtering untuk meraih goal pertama dan Collaborative Based Filtering untuk goal yang kedua. Ide dari Content Based Filtering adalah merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu. Hasil rekomendasi dari model ini akan bersifat subjektif karena melihat sejarah pengguna. Model content based akan dibuat dengan TF-IDF Vectorizer dan Cosine Similarity.

- Menggunakan Collaborative Based Filtering untuk meraih goal kedua. Collaborative Based Filtering bergantung pada pendapat komunitas pengguna. Ia tidak memerlukan atribut untuk setiap itemnya seperti pada sistem berbasis konten. Hasil rekomendasi dari model ini akan bersifat objektif karena dia menggunakan informasi pengguna lain. Model ini akan dibuat menggunakan RecommenderNet.

## Data Understanding

Dataset yang saya gunakan bernama "Anime Recommendations Database". Dataframe anime.csv memiliki sekitar 12,300 jumlah data. Dataframe rating.csv memiliki sekitar 7,81 juta jumlah data. Dataset yang saya gunakan bisa diunduh di tautan [ini](https://www.kaggle.com/CooperUnion/anime-recommendations-database). Di dalam dataset ini terdapat dua dataframe, yaitu:

**anime.csv**
- anime_id : id unik anime
- name : nama anime
- genre : genre anime
- type : cara anime disiarkan (TV, Movie, etc)
- episodes : jumlah episode anime
- rating : rating anime
- members : jumlah user dalam komunitas anime tersebut

![anime.csv](https://raw.githubusercontent.com/tutuna9/Dicoding-Predictive-Analytics/main/Aset%202/anime.csv.png)

**rating.csv**
- user_id : id unik user
- anime_id : id unik anime
- rating : rating yang diberikan user

![rating.csv](https://raw.githubusercontent.com/tutuna9/Dicoding-Predictive-Analytics/main/Aset%202/rating.csv.png)

## Data Preparation

### Data Cleaning

- Saya membuang nilai null dari anime.csv dan rating.csv untuk menghindari error dalam membuat model

Khusus untuk Collaborative Based Filtering (dalam rating.csv):

### Data Transform

- Menyandikan (encode) fitur ‘user_id’ dan ‘anime_id’ ke dalam indeks integer. Hal ini dilakukan karena model machine learning tidak bisa menerima data tipe objek, sehingga kita mengubahnya ke numerik. 
- Memetakan ‘user_id’ dan ‘anime_id’ ke dataframe yang berkaitan. Hal ini dilakukan untuk memberikan informasi yang jelas dan akan mudah jika kita ingin memakai informasi ini.
- Melakukan proses normalisasi terhadap nilai rating. Hal ini dilakukan agar hasil model kita nanti akan lebih akurat.

### Feature Engineering

- Membagi Data untuk Training dan Validasi untuk Collaborative Based Filtering. Hal ini dilakukan agar model kita menghindari masalah seperti overfitting dan underfitting.

## Modeling

1. **Content Based Filtering**
    Pada content Based Filtering, saya menggunakan TF-IDF Vectorizer untuk membangun sistem rekomendasi berdasarkan genre anime. Alasannya adalah untuk menemukan representasi fitur penting dari setiap genre anime. Lalu, saya ubah vektor tf-idf dalam bentuk matriks dengan fungsi todense(). Setelah itu, saya menghitung derajat kesamaan (similarity degree) antar anime dengan teknik cosine similarity. Terakhir, saya membuat fungsi anime_recommendations dengan beberapa parameter sebagai berikut:
    - nama_anime : Nama anime
    - similarity_data : Dataframe mengenai similarity yang telah kita definisikan sebelumnya
    - items : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan
    - k : Banyak rekomendasi yang ingin diberikan
    
    Ini hasilnya ketika saya menggunakan fungsinya:

![Hasil Content Based](https://raw.githubusercontent.com/tutuna9/Dicoding-Predictive-Analytics/main/Aset%202/Content%20Based.png)
    Bisa dilihat anime yang direkomendasikan memiliki genre yang sama dengan anime yang saya pilih.

2. **Collaborative Based Filtering**
    Pada model ini, saya menggunakan RecommenderNet. Setelah itu saya me-compile model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. Lalu, saya melatih model dengan batch size = 8, dan epoch = 100. Untuk mendapatkan rekomendasi, saya membuat fungsi untuk mendapatkan anime yang belum ditonton oleh user tersebut dengan menyocokkan anime_id yang berada di anime.csv dan rating.csv . Ini hasilnya ketika saya menggunakan fungsinya:

![Collaborative Based](https://raw.githubusercontent.com/tutuna9/Dicoding-Predictive-Analytics/main/Aset%202/Collaborative%20Based%20(2).png)

Bisa dilihat genre anime yang direkomendasikan memiliki persamaan dengan anime yang sudah ditonton oleh user.
    
## Evaluation

### Content Based Filtering

Pada Content Based Filtering, saya mencoba mengevaluasi model saya dengan memakai metrik precision. Maksud dari precision di sini adalah, berapa banyak genre yang sesuai dengan anime yang dipilih / jumlah rekomendasi. Saya membuat sebuah if loop yang akan membuat variabel 'a' bertambah satu jika genre sama persis dengan anime yang dipilih. Kodenya bisa dilihat seperti ini:

![Precision](https://raw.githubusercontent.com/tutuna9/Dicoding-Predictive-Analytics/main/Aset%202/Precision%20(2).png)

Itu saya menggunakan anime 'Kimi no Na Wa.' sebagai contoh. Dari gambar itu, artinya dua dari lima anime yang direkomendasikan memiliki genre yang sama persis dengan anime tersebut sehingga menghasilkan precision sebesar 40%.

### Collaborative Based Filtering

Pada Collaborative Based Filtering, saya memakai metrik evaluasi RMSE untuk mengevaluasi model saya. RMSE adalah aturan penilaian kuadrat yang mengukur besarnya rata-rata kesalahan. Ini adalah akar kuadrat dari rata-rata perbedaan kuadrat antara prediksi dan observasi aktual. Ini adalah rumus RMSE:

![Rumus RMSE](https://raw.githubusercontent.com/tutuna9/Dicoding-Predictive-Analytics/main/Aset%202/rumus%20rmse.jpg)

dimana:

At = Nilai data Aktual
Ft = Nilai hasil peramalan
N= banyaknya data
∑ = Summation (Jumlahkan keseluruhan  nilai)

Lalu, ini adalah visualisasi proses training melalui plot metrik evaluasi

![Grafik](https://raw.githubusercontent.com/tutuna9/Dicoding-Predictive-Analytics/main/Aset%202/Evaluasi%20Collaborative%20(2).png)

Bisa dilihat untuk RMSE train nilainya dibawah 0,1. Sedangkan untuk test, nilainya disekitar 0,2


