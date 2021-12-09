# Laporan Proyek Machine Learning - M. Ichsan Nur Iman

## Domain Proyek

Di zaman sekarang, kesehatan adalah salah satu hal yang paling diutamakan oleh orang. Dengan bantuan asuransi kesehatan, semua orang bisa mendapatkan layanan yang memadai. Dengan membuat predictive analysis terhadap dataset ini, kita bisa mengetahui kelompok apa yang lebih membutuhkan asuransi.

Berdasarkan [US Census Bureau](https://www.census.gov/library/publications/2021/demo/p60-274.html) pada tahun 2020, 8,6% penduduk, atau 28 juta, tidak memiliki asuransi kesehatan di tahun itu. Mungkin kita bisa menggunakan hasil analisis ini untuk membuat mereka mempunyai asuransi kesehatan.

## Business Understanding

### Problem Statement:
- Fitur variable apa yang paling mempengaruhi fitur target(biaya asuransi)?
- Model machine learning apa yang paling baik untuk memprediksi biaya asuransi?

### Goals:
- Mengetahui fitur variable apa yang paling mempengaruhi fitur target(biaya asuransi)
- Membuat model machine learning yang paling tepat untuk memprediksi biaya asuransi

### Solution Statements:
- Menggunakan EDA untuk mencari tahu fitur apa yang berkolerasi dengan kenaikan biaya asuransi.
- Membuat tiga model machine learning dengan pendekatan yang berbeda. Kemudian kita akan memilih model mana yang memiliki tingkat akurasi paling tinggi, pendekatan yang akan digunakan adalah:
	1.	K-Nearest Neighbor
	2.	Random Forest
	3.	Boosting Algorithm

## Data Understanding

Dataset yang saya gunakan bernama "US Health Insurance Dataset". Dataset ini dapat didownload di [sini](https://www.kaggle.com/teertha/ushealthinsurancedataset). Dataset ini memiliki 1338 baris data dan tidak ada bagian yang null di dalamnya. Di dalam dataset ini, ada 7 variabel dengan 4 fitur dan 3 label. Berikut adalah variable dengan penjelasannya:
- 	Age : Umur pengguna asuransi
-	Sex : Jenis kelamin pengguna asuransi
-	BMI : Indeks Massa Tubuh pengguna asuransi
-	Children : Jumlah anak yang dimiliki pengguna asuransi
-	Smoker : Apakah pengguna asuransi merokok atau tidak
-	Region : Daerah tempat tinggal pengguna asuransi
-	Charges : Biaya pengguna asuransi

Berikut adalah tahapan yang saya lakukan untuk memahami data:

### Analisis Data

-	Univariate analysis
 
	Ada 3 barplot yang saya buat di notebook. 

-   Barplot Pertama
   
![Barplot Pertama](https://raw.githubusercontent.com/tutuna9/Dicoding-Predictive-Analytics/main/Sex.png)

Kita bisa lihat jumlah pria dan wanita hampir sama banyak

-  Barplot Kedua

![Barplot kedua](https://raw.githubusercontent.com/tutuna9/Dicoding-Predictive-Analytics/main/Smoker.png))

Dari barplot ini, kita lihat jumlah perokok lebih sedikit dari yang tidak merokok

-  Barplot Ketiga
  
![Barplot Ketiga](https://raw.githubusercontent.com/tutuna9/Dicoding-Predictive-Analytics/main/Region.png))

Jika kita lihat, jumlah dari masing-masing keempat daerah rata-rata sama banyaknya

Saya juga menggunakan histogram untuk menulusuri data

![Histogram](https://raw.githubusercontent.com/tutuna9/Dicoding-Predictive-Analytics/main/Histogram.png)

-	Multivariate Analysis

Untuk multivariate analysis, saya menggunakan barplot, pairplot, dan heatmap.

- Bar Plot
 
![Barplot](https://raw.githubusercontent.com/tutuna9/Dicoding-Predictive-Analytics/main/barplot.png)

Dari bar plot ini, informasi yang paling jelas adalah bahwa orang yang merokok membayar biaya asuransi yang lebih besar

- Pair Plot

![Pairplot](https://raw.githubusercontent.com/tutuna9/Dicoding-Predictive-Analytics/main/Pairplot.png)

Dari plot ini, kita lihat hanya umur yang berkolerasi dengan biaya asuransi. Semakin tua seseorang, semakan besar biayanya.

- Heat Map

![Heatmap](https://raw.githubusercontent.com/tutuna9/Dicoding-Predictive-Analytics/main/Heatmap.png)

Ini kolerasi dilihat dari heat map. Kita lihat umur memiliki kolerasi yang paling besar dengan biaya asuransi dan jumlah anak kurang berpengaruh terhadap perkiraan biaya asuransi, jadi fitur tersebut dapat di drop

## Data Preparation

### Feature Selection

- Saya membuang fitur 'children' dikarenakan fitur itu memiliki pengaruh yang kecil terhadap fitur target.

### Data Transform

-	Encoding Fitur Kategori
	Saya menggunakan teknik one-hot-encoding kepada variabel kategori, seperti 'sex', 'smoker', 'region'. Hal ini dilakukan karena model machine learning tidak bisa menerima data tipe objek, sehingga kita mengubahnya ke numerik.

-	Train-Test-Split
	Membagi dataset menjadi data latih (train) dan data uji (test). Hal ini dilakukan agar model kita menghindari masalah seperti overfitting dan underfitting.

-	Standarisasi
	Standarisasi dilakukan untuk mengubah data kita memiliki skala relatif sama atau mendekati distribusi normal. Standarisasi digunakan untuk variabel numerik. Kita melakukan standarisasi agar algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat.

## Modeling

Inilah pendekatan yang saya gunakan:

1.	**K-Nearest Neighbor**
    Algoritma ini menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat. Kelebihan algoritma ini adalah algoritma ini relatif sederhana dibandingkan dengan algoritma lain. Kekurangannya adalah jika algoritma ini digunakan terhadap data dengan jumlah fitur atau dimensi yang besar. Parameter yang saya gunakan adalah 'n_neighbors'. Fungsi parameter ini adalah untuk menentukan jumlah tetangga yang dibandingkan.


2.	**Random Forest**
    Algoritma ini merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Kelebihan dari algoritma ini adalah mereka bekerja sama menyelesaikan masalah, sehingga tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Kekurangannya adalah waktu yang dibutuhkan lebih lama karena menggunakan banyak model dan menggabungkan semua model di akhir. Parameter yang saya gunakan adalah:

    - n_estimators = berapa jumlah pohon dalam hutan
    - max_depth = kedalaman maksimum pohon
    - random_state = generator angka acak
    - n_jobs = jumlah pekerjaan yang dijalankan secara paralel


3. **Boosting Algorithm**
    Algoritman ini juga termasuk dalam ensemble learning. Algoritma boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Oleh karena itu, model akhirnya akan sangat kuat. Kekurangan dari algoritama ini adalah algortima ini membutuhkan dataset yang berkualitas. Parameter yang saya gunakan adalah:

    - n_estimators = jumlah maksimal estimasi dimana boosting dihentikan
    - learning_rate = bobot untuk setiap iterasi boosting
    - random_state = generator angka acak

Setelah dievaluasi, algoritma Random Forest adalah pendekatan yang terbaik karena nilai MSE-nya paling kecil.

## Evaluasi

Metrik evaluasi yang digunakan adalah MSE(Mean Squared Error) yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Nilai MSE yang rendah menunjukkan hasil prediksi mendekati data aktual.

Berikut adalah hasil evaluasi dari ketiga model:

![Evaluasi](https://raw.githubusercontent.com/tutuna9/Dicoding-Predictive-Analytics/main/Evaluasi%20(2).png)

Bisa dilihat algoritma Random Forest memiliki nilai error yang paling kecil.

![tabel_evaluasi](https://raw.githubusercontent.com/tutuna9/Dicoding-Predictive-Analytics/main/Tabel%20Evaluasi%20(3).png)

Ini hasilnya ketika kita memprediksi menggunakan beberapa harga dari data test. Algoritma Random Forest memiliki tujuh dari sepuluh hasil yang paling mendekati dengan data.
 

