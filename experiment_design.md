# Experiment Design: Perbandingan Pendekatan ABSA untuk Ulasan Perhotelan

**Versi:** 1.0\
**Tujuan dokumen:** menjadi spesifikasi eksperimen dan acuan
implementasi kode untuk membandingkan pendekatan leksikon, machine
learning konvensional, dan transformer pada tugas Aspect-Based Sentiment
Analysis (ABSA) ulasan perhotelan.

------------------------------------------------------------------------

## 1. Tujuan dan ruang lingkup eksperimen

Eksperimen bertujuan membandingkan tiga pendekatan untuk memprediksi
polaritas sentimen pada setiap aspek dalam ulasan hotel:

1.  **Lexicon-based:** InSet (Indonesian Sentiment Lexicon) yang
    diperluas dengan kosakata domain perhotelan dan aturan berbasis
    leksikon.
2.  **Conventional machine learning:** TF-IDF sebagai representasi teks
    dan Support Vector Machine (SVM) sebagai classifier.
3.  **Transformer-based:** IndoBERT-base-p2 yang di-fine-tune untuk
    klasifikasi sentimen berbasis aspek.

Ketiga pendekatan harus dievaluasi menggunakan **data uji yang sama**.
Model terbaik dipilih berdasarkan performa pada data uji dan hasil
pengujian signifikansi statistik, kemudian digunakan untuk memproses
seluruh dataset ulasan yang telah dikumpulkan.

### 1.1 Pertanyaan penelitian

-   RQ1: Bagaimana performa pendekatan leksikon, SVM berbasis TF-IDF,
    dan IndoBERT dalam memprediksi polaritas sentimen pada setiap aspek
    ulasan hotel?
-   RQ2: Apakah perbedaan performa antarmodel pada data uji signifikan
    secara statistik?
-   RQ3: Pendekatan mana yang memberikan performa terbaik dan paling
    layak digunakan untuk memproses seluruh dataset ulasan?

### 1.2 Unit analisis

Unit analisis yang direkomendasikan adalah **pasangan ulasan-aspek**
(*review-aspect pair*), bukan hanya satu ulasan utuh. Satu ulasan dapat
menghasilkan beberapa baris apabila ulasan tersebut membahas beberapa
aspek.

Contoh:

  -----------------------------------------------------------------------
  review_id         review_text       aspect            sentiment
  ----------------- ----------------- ----------------- -----------------
  R001              Kamar bersih,     kebersihan kamar  positif
                    tetapi sarapannya                   
                    kurang enak.                        

  R001              Kamar bersih,     makanan/sarapan   negatif
                    tetapi sarapannya                   
                    kurang enak.                        
  -----------------------------------------------------------------------

Dengan format ini, ketiga pendekatan mengerjakan target prediksi yang
sama: **memprediksi polaritas untuk aspek yang diberikan**.

> **Keputusan desain penting:** flowchart menunjukkan adanya ekstraksi
> aspek pada pendekatan leksikon, sedangkan alur SVM dan IndoBERT lebih
> menonjolkan klasifikasi sentimen. Agar perbandingan adil, ketiga model
> harus menerima informasi aspek yang sama dan dievaluasi pada pasangan
> ulasan-aspek yang sama. Dokumen ini menetapkan klasifikasi polaritas
> berbasis aspek sebagai tugas utama. Jika sistem juga harus menemukan
> aspek secara otomatis dari teks mentah, tahap ekstraksi aspek perlu
> dievaluasi sebagai tugas terpisah atau diintegrasikan secara konsisten
> ke ketiga pendekatan.

------------------------------------------------------------------------

## 2. Definisi tugas dan label

### 2.1 Label polaritas

Gunakan definisi label yang konsisten dengan pedoman anotasi:

-   `positive`: ulasan menyatakan penilaian positif terhadap aspek.
-   `negative`: ulasan menyatakan penilaian negatif terhadap aspek.
-   `neutral`: ulasan menyebut aspek secara informatif atau faktual
    tanpa evaluasi positif maupun negatif.

Jika pedoman anotasi menggunakan label lain, misalnya `netral` atau
`mixed`, sesuaikan pemetaan label sebelum eksperimen dimulai. Jangan
menggabungkan atau mengubah label setelah melihat hasil data uji.

**Rekomendasi:** gunakan tiga kelas (positive, negative, neutral) hanya
apabila ketiganya memang tersedia dan didefinisikan dalam anotasi. Jika
anotasi hanya memiliki dua kelas, seluruh konfigurasi model dan metrik
harus menggunakan dua kelas tersebut.

### 2.2 Daftar aspek

Tetapkan daftar kategori aspek yang menjadi target eksperimen
berdasarkan codebook anotasi. Contoh kategori domain perhotelan (bukan
daftar final):

-   kebersihan kamar
-   fasilitas kamar
-   kenyamanan
-   pelayanan staf
-   lokasi
-   makanan/sarapan
-   harga
-   fasilitas umum

Nama aspek harus dinormalisasi menjadi label kanonis. Contohnya, istilah
"room cleanliness" dan "kebersihan kamar" tidak boleh diperlakukan
sebagai dua kategori berbeda apabila maknanya sama.

### 2.3 Aturan untuk kasus khusus

Tentukan sebelum eksperimen bagaimana anotasi menangani:

-   ulasan yang tidak menyebut aspek tertentu;
-   satu aspek yang memiliki sentimen berbeda dalam satu ulasan;
-   kalimat yang mengandung negasi;
-   sarkasme atau ironi;
-   sentimen campuran;
-   ungkapan yang tidak cukup informatif;
-   aspek yang disebut secara implisit.

Jangan membuat pasangan ulasan-aspek untuk aspek yang tidak dibahas,
kecuali pedoman anotasi memang menetapkan label khusus untuk kasus
tersebut.

------------------------------------------------------------------------

## 3. Dataset dan pembagian data

### 3.1 Sumber data

Gunakan dataset ulasan hotel yang telah melalui proses anotasi. Minimal
setiap observasi harus memiliki:

-   `review_id`: identitas unik ulasan;
-   `review_text`: teks ulasan asli;
-   `aspect`: kategori aspek;
-   `sentiment`: label polaritas untuk aspek tersebut.

Simpan teks asli dan hasil preprocessing pada kolom terpisah. Teks asli
tidak boleh ditimpa agar proses eksperimen dapat ditelusuri.

### 3.2 Proporsi pembagian

Dataset anotasi dibagi menjadi:

  ------------------------------------------------------------------------
  Subset                                    Proporsi Fungsi
  --------------------- ---------------------------- ---------------------
  Data latih                                     70% Membangun model dan
                                                     mempelajari parameter
                                                     dari data

  Data validasi                                  15% Memilih konfigurasi
                                                     dan hyperparameter

  Data uji                                       15% Evaluasi final
                                                     setelah konfigurasi
                                                     ditetapkan
  ------------------------------------------------------------------------

Pembagian dilakukan **sebelum** pelatihan model dan preprocessing yang
mempelajari statistik dari korpus.

### 3.3 Pencegahan data leakage

Pembagian data harus mencegah informasi dari ulasan yang sama muncul
pada subset berbeda.

-   Lakukan split berdasarkan `review_id`, bukan berdasarkan baris
    pasangan ulasan-aspek.
-   Jika satu ulasan memiliki lima aspek, seluruh pasangan dari ulasan
    tersebut harus berada pada subset yang sama.
-   Jika tersedia beberapa ulasan yang merupakan duplikasi atau salinan
    hampir identik, identifikasi dan kelompokkan sebelum split.
-   Gunakan stratifikasi berdasarkan label sentimen sejauh memungkinkan,
    dengan tetap mempertahankan pengelompokan berdasarkan `review_id`.
-   Jika stratifikasi gabungan aspek dan sentimen menyebabkan kelompok
    terlalu kecil, prioritaskan pencegahan leakage dan dokumentasikan
    distribusi aktual.

Simpan daftar `review_id` untuk setiap subset agar split yang sama dapat
digunakan oleh seluruh pendekatan.

### 3.4 Pemeriksaan distribusi

Sebelum training, buat tabel distribusi berikut untuk setiap subset:

1.  jumlah ulasan unik;
2.  jumlah pasangan ulasan-aspek;
3.  jumlah sampel per aspek;
4.  jumlah sampel per label sentimen;
5.  distribusi label untuk setiap aspek.

Periksa apakah ada kelas yang tidak muncul pada data latih atau
validasi. Kondisi tersebut harus diselesaikan atau dicatat sebelum
eksperimen dijalankan.

### 3.5 Artefak pembagian data

Simpan sekurang-kurangnya:

-   `train.csv`
-   `validation.csv`
-   `test.csv`
-   `split_manifest.json` yang berisi seed, proporsi, dan daftar ID pada
    masing-masing subset.

Data uji tidak boleh digunakan untuk memilih kosakata, threshold,
hyperparameter, epoch, atau konfigurasi model.

------------------------------------------------------------------------

## 4. Persiapan data dan preprocessing

### 4.1 Prinsip umum

-   Pertahankan teks asli sebagai sumber utama.
-   Terapkan normalisasi yang konsisten dan dapat direproduksi.
-   Jangan menggunakan informasi label sentimen untuk mengubah teks.
-   Jangan melakukan oversampling sebelum pembagian data.
-   Jangan menggabungkan data validasi atau uji ke data latih selama
    pemilihan model.

### 4.2 Normalisasi yang disarankan

Buat fungsi preprocessing yang terdokumentasi, misalnya:

1.  normalisasi encoding dan karakter Unicode;
2.  penghapusan spasi berlebih;
3.  normalisasi URL dan mention jika relevan;
4.  penanganan karakter berulang secara konsisten;
5.  normalisasi slang atau singkatan menggunakan kamus yang ditetapkan
    sebelumnya.

Jangan menghapus kata negasi seperti `tidak`, `bukan`, `kurang`,
`belum`, atau `tanpa`. Kata tersebut dapat membalik polaritas sentimen.

Jangan menghapus kata sifat, kata evaluatif, angka, atau kata yang
menunjukkan intensitas secara sembarangan. Kata seperti "sangat",
"cukup", "terlalu", dan "agak" dapat memengaruhi polaritas atau
intensitas.

### 4.3 Preprocessing per pendekatan

Preprocessing tidak harus identik apabila kebutuhan algoritmanya
berbeda, tetapi seluruh perbedaan harus didokumentasikan.

  -----------------------------------------------------------------------
  Komponen          Leksikon          SVM + TF-IDF      IndoBERT
  ----------------- ----------------- ----------------- -----------------
  Lowercase         Ya, jika kamus    Ya, sebagai       Ikuti
                    menggunakan       konfigurasi awal  tokenizer/model
                    bentuk lowercase                    

  Stopword removal  Hindari menghapus Uji tanpa         Tidak dilakukan
                    negasi dan kata   stopword removal  
                    evaluatif         sebagai baseline  

  Stemming          Opsional;         Opsional;         Tidak dilakukan
                    evaluasi          tetapkan sebelum  
                    dampaknya         tuning            

  Tokenisasi        Tokenisasi        Tokenizer TF-IDF  Tokenizer bawaan
                    sederhana untuk                     checkpoint
                    pencocokan kamus                    

  Negasi            Ditangani oleh    Dipertahankan     Dipertahankan
                    aturan leksikon   sebagai fitur     dalam input
                                      teks              
  -----------------------------------------------------------------------

**Penting:** jangan menerapkan stemming atau stopword removal secara
otomatis pada IndoBERT. Model menggunakan tokenizer dan representasi
subword yang telah dipelajari saat pre-training.

------------------------------------------------------------------------

## 5. Pendekatan 1 --- Lexicon-based menggunakan InSet

### 5.1 Tujuan

Membangun baseline berbasis aturan tanpa proses pelatihan classifier.
Polaritas ditentukan dari pencocokan kata atau frasa pada kamus sentimen
InSet yang diperluas dengan kosakata domain perhotelan.

### 5.2 Sumber leksikon

Gunakan:

1.  InSet sebagai leksikon sentimen Bahasa Indonesia dasar.
2.  Ekstensi kosakata domain perhotelan yang relevan dengan ulasan
    hotel.

Leksikon perlu memiliki setidaknya kolom:

  token            polarity_score source         notes
  -------------- ---------------- -------------- --------
  bersih                       +1 InSet/domain   contoh
  nyaman                       +1 InSet/domain   contoh
  kotor                        -1 InSet/domain   contoh
  mengecewakan                 -1 InSet/domain   contoh

Nilai pada tabel hanya ilustrasi. Gunakan skor aktual dari sumber
leksikon atau skema skor yang ditetapkan dalam penelitian.

Untuk setiap entri ekstensi, catat asal kata, alasan penambahan, dan
polaritas yang ditetapkan. Hindari memasukkan kosakata berdasarkan hasil
pada data uji.

### 5.3 Penyusunan ekstensi domain

-   Identifikasi istilah yang sering muncul dalam konteks perhotelan.
-   Kelompokkan sinonim, variasi ejaan, dan istilah domain.
-   Tetapkan polaritas berdasarkan pedoman anotasi dan penilaian
    ahli/domain expert.
-   Dokumentasikan kata yang ambigu atau bergantung konteks.
-   Bekukan versi kamus sebelum evaluasi data uji.

Contoh istilah domain yang perlu ditinjau: "check-in", "resepsionis",
"AC", "sprei", "sarapan", "parkir", "kolam renang", "air panas", dan
"akses lokasi". Istilah tersebut tidak otomatis positif atau negatif;
polaritas ditentukan oleh konteks atau frasa evaluatif yang
menyertainya.

### 5.4 Pencocokan aspek

Prediksi harus spesifik terhadap aspek yang sedang dinilai.

Rekomendasi implementasi:

1.  normalisasi teks ulasan;
2.  identifikasi istilah/frasa yang berhubungan dengan aspek target
    menggunakan kamus aspek atau aturan pencocokan;
3.  ambil konteks lokal di sekitar istilah aspek;
4.  cari kata/frasa sentimen pada konteks tersebut;
5.  hitung skor polaritas berdasarkan entri leksikon yang cocok.

Jika anotasi menyediakan target aspek dalam bentuk kategori, gunakan
kategori tersebut sebagai input untuk memilih kamus aspek dan konteks.
Hindari menghitung semua kata sentimen dalam satu ulasan untuk setiap
aspek, karena hal itu dapat mengatribusikan sentimen aspek A ke aspek B.

Jika aspek disebut secara implisit atau tidak ditemukan oleh aturan,
tetapkan fallback yang konsisten, misalnya `neutral` atau
`no_prediction`. Pilihan ini harus dinyatakan sejak awal dan dievaluasi
secara konsisten.

### 5.5 Aturan polaritas

Rancang aturan minimum berikut:

-   **Negasi:** kata seperti "tidak", "bukan", "kurang", dan "belum"
    dapat membalik atau memodifikasi polaritas kata setelahnya dalam
    jendela konteks tertentu.
-   **Intensifier:** kata seperti "sangat", "amat", atau "terlalu" dapat
    memperkuat skor.
-   **Diminisher:** kata seperti "agak" atau "cukup" dapat mengurangi
    intensitas skor.
-   **Konjungsi/kontras:** kata seperti "tetapi", "namun", dan
    "walaupun" dapat menandai perubahan opini dalam kalimat.
-   **Frasa multi-kata:** pencocokan frasa harus diprioritaskan terhadap
    pencocokan token tunggal jika terdapat entri frasa.

Jendela negasi dan intensifier harus ditetapkan secara eksplisit,
misalnya sejumlah token sebelum kata sentimen. Jangan mengklaim aturan
tersebut memahami seluruh konteks bahasa; catat sebagai keterbatasan
pendekatan berbasis aturan.

### 5.6 Skor dan keputusan kelas

Untuk setiap pasangan ulasan-aspek, hitung skor:

`score = jumlah skor polaritas setelah menerapkan aturan`

Kemudian tetapkan ambang keputusan. Contoh skema sederhana:

-   `score > 0` → positive
-   `score < 0` → negative
-   `score = 0` → neutral

Jika menggunakan threshold lain, tentukan hanya melalui data
latih/validasi. Jangan memilih threshold berdasarkan data uji.

Untuk kasus tanpa kecocokan leksikon, tentukan apakah hasilnya `neutral`
atau `no_prediction`. Jika `no_prediction` digunakan, tetapkan bagaimana
hasil tersebut dihitung dalam evaluasi; jangan diam-diam menghapus
sampel yang gagal diprediksi.

### 5.7 Evaluasi pendekatan leksikon

Pendekatan ini tidak memiliki training atau hyperparameter classifier.
Namun, aturan dan threshold dapat ditetapkan melalui data latih dan
validasi.

-   Data latih: pengembangan dan pemeriksaan kamus/aturan.
-   Data validasi: pemilihan konfigurasi aturan atau threshold.
-   Data uji: evaluasi final satu kali setelah kamus dan aturan
    dibekukan.

**Artefak yang disimpan:**

-   file leksikon dasar dan ekstensi domain;
-   versi kamus aspek;
-   file aturan negasi/intensifier;
-   konfigurasi threshold;
-   kode prediksi;
-   prediksi data validasi dan data uji.

------------------------------------------------------------------------

## 6. Pendekatan 2 --- Conventional ML menggunakan TF-IDF + SVM

### 6.1 Tujuan

Membangun classifier konvensional yang memprediksi polaritas sentimen
berdasarkan representasi TF-IDF dari teks ulasan dan informasi aspek.

### 6.2 Formulasi input

Gunakan pasangan ulasan-aspek sebagai satu sampel. Bentuk input yang
direkomendasikan:

`[ASPEK] kebersihan kamar [ULASAN] kamar sangat bersih dan nyaman`

Dengan demikian, classifier dapat mempelajari hubungan antara konteks
ulasan dan aspek yang sedang dinilai.

Jika implementasi memilih classifier terpisah untuk setiap aspek,
dokumentasikan keputusan tersebut dan pastikan evaluasi tetap dilakukan
pada kumpulan pasangan ulasan-aspek yang sama. Rekomendasi utama dokumen
ini adalah satu classifier multi-kelas dengan aspek sebagai bagian
input.

### 6.3 Ekstraksi fitur TF-IDF

Gunakan `TfidfVectorizer` dengan konfigurasi awal yang dapat diuji:

-   `ngram_range`: `(1, 1)` dan `(1, 2)`;
-   `min_df`: beberapa kandidat kecil yang sesuai ukuran korpus;
-   `max_df`: kandidat seperti `0.9` atau `1.0`;
-   `sublinear_tf`: `True` atau `False`;
-   `max_features`: tanpa batas atau batas yang ditentukan berdasarkan
    ukuran data.

Daftar konfigurasi tersebut merupakan kandidat eksperimen, bukan hasil
yang harus diasumsikan terbaik.

**Aturan penting:** `fit` vectorizer hanya menggunakan teks data latih.
Data validasi dan data uji hanya menggunakan `transform` dari vectorizer
yang telah di-fit pada data latih. Jangan melakukan `fit_transform` pada
seluruh dataset.

Gunakan pipeline agar vectorizer dan classifier selalu terlatih sebagai
satu kesatuan dan mencegah kebocoran data.

### 6.4 Classifier SVM

Gunakan Support Vector Machine dengan `SVC` sebagai konfigurasi utama.

Parameter yang dapat diuji:

  Parameter        Kandidat
  ---------------- --------------------
  `kernel`         `linear`, `rbf`
  `C`              `0.1`, `1`, `10`
  `class_weight`   `None`, `balanced`

Mulai dari kernel linear sebagai baseline karena TF-IDF umumnya
menghasilkan fitur sparse berdimensi tinggi. Kernel RBF dapat diuji jika
sumber daya dan ukuran data memungkinkan.

Jika menggunakan `LinearSVC`, dokumentasikan bahwa classifier yang
digunakan berbeda dari `SVC(kernel="linear")`. Jangan mencampur hasil
kedua implementasi tanpa mencatat konfigurasi yang tepat.

### 6.5 Naive Bayes sebagai pembanding opsional

Flowchart mencantumkan SVM/Naive Bayes. Jika penelitian hanya menetapkan
SVM, Naive Bayes tidak wajib dijalankan. Jika ingin mempertahankan kedua
classifier, perlakukan keduanya sebagai model eksperimen terpisah:

-   TF-IDF + SVM
-   TF-IDF + Multinomial Naive Bayes

Jangan menggabungkan hasil SVM dan Naive Bayes menjadi satu model.
Perbandingan utama tiga pendekatan pada dokumen ini menggunakan SVM.
Naive Bayes dapat menjadi eksperimen tambahan.

### 6.6 Prosedur pelatihan dan tuning

1.  Fit TF-IDF pada data latih.
2.  Transform data latih dan validasi.
3.  Latih SVM menggunakan label data latih.
4.  Prediksi data validasi.
5.  Hitung metrik validasi.
6.  Ulangi untuk kombinasi hyperparameter yang ditentukan.
7.  Pilih konfigurasi berdasarkan metrik utama yang ditetapkan.
8.  Simpan konfigurasi dan model terpilih.
9.  Jalankan prediksi final pada data uji.

Jangan menggunakan data uji untuk memilih `C`, kernel, n-gram,
threshold, atau parameter TF-IDF.

### 6.7 Pemilihan model SVM

Gunakan **Macro-F1** sebagai metrik utama pemilihan konfigurasi apabila
distribusi kelas tidak seimbang. Jika penelitian menetapkan metrik utama
lain, nyatakan sebelum eksperimen dan gunakan secara konsisten.

Jika dua konfigurasi memiliki nilai yang sama atau sangat dekat, gunakan
aturan tie-break yang ditetapkan sebelumnya, misalnya memilih model yang
lebih sederhana.

### 6.8 Artefak yang disimpan

-   konfigurasi TF-IDF;
-   vocabulary dan IDF hasil fitting;
-   parameter SVM;
-   model/pipeline terlatih;
-   skor setiap konfigurasi pada validasi;
-   prediksi validasi dan uji;
-   random seed dan versi library.

------------------------------------------------------------------------

## 7. Pendekatan 3 --- IndoBERT-base-p2 fine-tuning

### 7.1 Tujuan

Mengembangkan model klasifikasi sentimen berbasis aspek dengan melakukan
fine-tuning terhadap checkpoint IndoBERT-base-p2 yang telah melalui
pre-training untuk Bahasa Indonesia.

Checkpoint yang digunakan harus dicatat secara spesifik, termasuk nama
repository/model dan revisi atau versi checkpoint jika tersedia. Jangan
hanya mencatat nama keluarga model.

### 7.2 Formulasi input

Gunakan format pasangan ulasan-aspek yang sama dengan pendekatan SVM:

`[ASPEK] kebersihan kamar [ULASAN] kamar sangat bersih dan nyaman`

Format dapat direalisasikan sebagai satu sequence teks atau sebagai
pasangan sequence, tetapi harus konsisten pada training, validasi, dan
inferensi.

Label target adalah polaritas sentimen untuk aspek yang diberikan.
Dengan demikian, model belajar memprediksi sentimen aspek, bukan
sentimen umum ulasan.

### 7.3 Tokenisasi

Gunakan tokenizer bawaan checkpoint IndoBERT-base-p2.

Parameter awal:

-   `max_length`: tentukan berdasarkan distribusi panjang teks; kandidat
    awal 128 atau 256 token;
-   `padding`: gunakan dynamic padding per batch atau padding sesuai
    konfigurasi;
-   `truncation`: aktifkan;
-   `return_tensors`: sesuai framework yang digunakan.

Periksa persentase teks yang terpotong. Jika banyak ulasan panjang
terpotong, evaluasi peningkatan `max_length` dengan mempertimbangkan
memori GPU dan waktu training.

Jangan menggunakan stemming, stopword removal, atau tokenisasi manual
sebelum tokenizer IndoBERT.

### 7.4 Arsitektur classifier

Gunakan IndoBERT sebagai encoder dan classification head untuk
memprediksi jumlah kelas polaritas yang ditetapkan pada Bagian 2.

Untuk klasifikasi sequence, representasi pooled/representasi token awal
sesuai implementasi model digunakan sebagai input classification head.
Catat implementasi model dan konfigurasi head yang digunakan.

### 7.5 Hyperparameter fine-tuning

Tetapkan ruang pencarian yang realistis dan dokumentasikan semua
kandidat.

  Hyperparameter        Kandidat awal
  --------------------- -------------------------------------
  Learning rate         `2e-5`, `3e-5`, `5e-5`
  Batch size            `8`, `16` (sesuaikan kapasitas GPU)
  Epoch                 `2`, `3`, `4`, `5`
  Weight decay          `0.01` atau nilai yang ditetapkan
  Warmup ratio          `0.0`, `0.1`
  Max sequence length   `128`, `256`

Nilai di atas adalah ruang pencarian awal, bukan konfigurasi terbaik
yang telah diketahui. Batasi jumlah kombinasi agar eksperimen tetap
terukur.

Gunakan optimizer AdamW sebagai konfigurasi awal. Catat scheduler,
warmup, gradient accumulation, gradient clipping, mixed precision, dan
konfigurasi GPU jika digunakan.

### 7.6 Prosedur fine-tuning

1.  Muat checkpoint IndoBERT-base-p2 dan tokenizer yang sesuai.
2.  Siapkan input pasangan ulasan-aspek untuk data latih dan validasi.
3.  Tokenisasi data menggunakan tokenizer checkpoint.
4.  Inisialisasi classification head sesuai jumlah kelas.
5.  Fine-tune model menggunakan data latih.
6.  Evaluasi pada data validasi di setiap epoch.
7.  Pilih checkpoint berdasarkan Macro-F1 validasi atau metrik utama
    yang telah ditetapkan.
8.  Gunakan early stopping bila diperlukan; tetapkan patience sebelum
    eksperimen.
9.  Simpan checkpoint terbaik dan konfigurasi training.
10. Jalankan prediksi final pada data uji.

Data uji tidak boleh digunakan untuk early stopping atau pemilihan
epoch.

### 7.7 Penanganan ketidakseimbangan kelas

Mulai dari loss standar. Jika distribusi kelas sangat tidak seimbang,
pertimbangkan class-weighted cross entropy yang bobotnya dihitung hanya
dari label data latih.

Jangan menggunakan oversampling pada data validasi atau uji. Jika
menggunakan weighted loss, dokumentasikan rumus dan bobot tiap kelas.

### 7.8 Reproducibility

Catat dan tetapkan seed untuk Python, NumPy, dan framework deep
learning. Catat versi:

-   Python;
-   PyTorch;
-   Transformers;
-   Tokenizers;
-   CUDA dan cuDNN;
-   GPU yang digunakan.

Hasil deep learning dapat sedikit berbeda akibat operasi
nondeterministik. Jika sumber daya memungkinkan, jalankan konfigurasi
final dengan beberapa seed dan laporkan variasi hasil. Jika hanya satu
seed digunakan, nyatakan sebagai keterbatasan.

### 7.9 Artefak yang disimpan

-   nama dan revisi checkpoint;
-   tokenizer;
-   konfigurasi model;
-   hyperparameter training;
-   log loss dan metrik per epoch;
-   checkpoint terbaik berdasarkan validasi;
-   prediksi validasi dan uji;
-   seed, versi library, dan informasi perangkat keras.

------------------------------------------------------------------------

## 8. Protokol evaluasi

### 8.1 Evaluasi pada data uji yang identik

Setelah setiap pendekatan selesai dituning, lakukan prediksi pada
**baris data uji yang sama**, dengan urutan dan identitas sampel yang
dapat dicocokkan.

Buat file prediksi dengan format minimal:

  ---------------------------------------------------------------------------------------------
  sample_id   review_id   aspect     y_true     y_pred_lexicon   y_pred_svm   y_pred_indobert
  ----------- ----------- ---------- ---------- ---------------- ------------ -----------------

  ---------------------------------------------------------------------------------------------

Pastikan seluruh model menghasilkan tepat satu prediksi untuk setiap
unit evaluasi, atau dokumentasikan secara eksplisit cara menangani
prediksi yang tidak tersedia.

### 8.2 Metrik evaluasi

Hitung metrik berikut:

-   **Accuracy:** proporsi prediksi yang benar dari seluruh sampel.
-   **Precision:** proporsi prediksi positif pada suatu kelas yang
    benar.
-   **Recall:** proporsi sampel aktual suatu kelas yang berhasil
    ditemukan.
-   **F1-score:** harmonic mean antara precision dan recall.

Untuk klasifikasi multi-kelas, laporkan precision, recall, dan F1 dengan
skema averaging yang eksplisit:

-   **Macro:** menghitung metrik per kelas lalu mengambil rata-rata
    tanpa bobot; penting untuk melihat performa kelas minoritas.
-   **Weighted:** rata-rata berbobot berdasarkan jumlah sampel tiap
    kelas.
-   **Per-class:** metrik untuk setiap label sentimen.

**Rekomendasi pelaporan:**

1.  Accuracy.
2.  Macro-Precision.
3.  Macro-Recall.
4.  Macro-F1.
5.  Weighted-F1 sebagai metrik tambahan.
6.  Precision, recall, dan F1 per kelas.
7.  Confusion matrix.

Gunakan Macro-F1 sebagai metrik utama pemilihan konfigurasi apabila
kelas tidak seimbang. Tetapkan keputusan ini sebelum melihat hasil uji.

### 8.3 Evaluasi per aspek

Karena tugas ABSA mengevaluasi polaritas pada setiap aspek, laporkan
metrik:

-   secara keseluruhan pada seluruh pasangan ulasan-aspek;
-   per kategori aspek.

Jangan hanya melaporkan skor gabungan. Skor gabungan dapat
menyembunyikan aspek yang memiliki performa rendah atau distribusi kelas
yang berbeda.

Jika suatu aspek memiliki jumlah sampel terlalu sedikit, berikan catatan
bahwa estimasi metriknya kurang stabil.

### 8.4 Confusion matrix dan analisis kesalahan

Buat confusion matrix untuk setiap model. Analisis jenis kesalahan,
misalnya:

-   positive diprediksi sebagai neutral;
-   negative diprediksi sebagai positive;
-   kesalahan akibat negasi;
-   kesalahan akibat sentimen campuran;
-   kesalahan atribusi sentimen ke aspek yang salah;
-   istilah domain atau slang yang tidak dikenali.

Ambil contoh kesalahan dari data uji untuk analisis kualitatif, tetapi
jangan gunakan contoh tersebut untuk mengubah model lalu melaporkan
hasil yang sama sebagai evaluasi final.

------------------------------------------------------------------------

## 9. Uji signifikansi: McNemar's test

### 9.1 Tujuan

McNemar's test digunakan untuk membandingkan dua classifier yang
memprediksi unit evaluasi yang sama, dengan melihat apakah pola
kesalahan keduanya berbeda secara signifikan.

Uji ini dilakukan terhadap hasil prediksi data uji, bukan skor agregat
seperti F1-score.

### 9.2 Perbandingan berpasangan

Lakukan tiga perbandingan:

1.  Lexicon vs SVM.
2.  Lexicon vs IndoBERT.
3.  SVM vs IndoBERT.

Untuk setiap pasangan model, buat tabel kontingensi berdasarkan
benar/salah:

                    Model B benar   Model B salah
  --------------- --------------- ---------------
  Model A benar                 a               b
  Model A salah                 c               d

McNemar's test terutama menggunakan pasangan diskordan `b` dan `c`,
yaitu sampel ketika hanya salah satu model yang benar.

### 9.3 Penerapan pada klasifikasi multi-kelas

Untuk setiap unit evaluasi, definisikan:

-   `correct_A = (y_pred_A == y_true)`
-   `correct_B = (y_pred_B == y_true)`

Kemudian jalankan McNemar's test terhadap dua vektor benar/salah
tersebut.

Pendekatan ini membandingkan tingkat kesalahan berpasangan pada tugas
multi-kelas dengan mengubah hasil setiap classifier menjadi benar atau
salah. Uji ini tidak secara khusus menguji perbedaan distribusi label
yang diprediksi.

### 9.4 Koreksi pengujian berganda

Karena terdapat tiga perbandingan berpasangan, kendalikan family-wise
error rate. Rekomendasi: gunakan koreksi Holm pada tiga p-value.

Tetapkan:

-   tingkat signifikansi awal `alpha = 0.05`;
-   jenis McNemar yang digunakan (exact binomial atau chi-square dengan
    continuity correction);
-   metode koreksi p-value (Holm).

Untuk jumlah pasangan diskordan yang kecil, exact McNemar lebih tepat
daripada pendekatan chi-square asimtotik.

Laporkan p-value mentah dan p-value setelah koreksi Holm.

### 9.5 Interpretasi

-   Jika p-value terkoreksi lebih kecil dari alpha, terdapat bukti
    perbedaan tingkat kesalahan yang signifikan antara dua model.
-   Jika p-value terkoreksi tidak lebih kecil dari alpha, belum terdapat
    bukti yang cukup untuk menyatakan perbedaan tingkat kesalahan
    signifikan.
-   Signifikansi statistik tidak otomatis berarti peningkatan performa
    tersebut besar atau penting secara praktis.

**Catatan unit analisis:** beberapa pasangan ulasan-aspek dapat berasal
dari `review_id` yang sama. McNemar standar mengasumsikan unit observasi
independen. Jika ketergantungan dalam satu ulasan cukup kuat, hasil
McNemar perlu ditafsirkan dengan hati-hati dan dapat dilengkapi analisis
berbasis resampling/cluster pada tingkat `review_id`.

------------------------------------------------------------------------

## 10. Pemilihan model terbaik

Pemilihan model dilakukan setelah:

1.  seluruh model menyelesaikan tuning menggunakan data validasi;
2.  konfigurasi model dibekukan;
3.  seluruh model menghasilkan prediksi pada data uji yang sama;
4.  metrik keseluruhan dan per aspek dihitung;
5.  McNemar's test dan koreksi Holm selesai.

Pertimbangkan secara bersama:

-   Macro-F1 sebagai metrik utama;
-   Accuracy, Macro-Precision, dan Macro-Recall;
-   performa per aspek;
-   confusion matrix dan pola kesalahan;
-   hasil signifikansi statistik;
-   biaya komputasi dan kebutuhan deployment.

Jangan memilih model hanya karena memiliki accuracy tertinggi. Jika
model dengan skor tertinggi tidak berbeda signifikan dari model lain,
nyatakan bahwa bukti statistik belum cukup untuk menyimpulkan
keunggulan. Dalam kondisi tersebut, efisiensi inferensi, ukuran model,
latensi, dan kemudahan pemeliharaan dapat menjadi pertimbangan tambahan
yang dilaporkan secara transparan.

------------------------------------------------------------------------

## 11. Inferensi pada seluruh dataset ulasan

Setelah model terbaik dipilih, gunakan model tersebut untuk memproses
seluruh dataset ulasan yang telah dikumpulkan.

### 11.1 Batasan penggunaan data

Data uji tetap menjadi subset evaluasi final dan tidak boleh dipakai
untuk tuning. Setelah evaluasi final dan keputusan model selesai, model
dapat dilatih ulang menggunakan data anotasi yang tersedia (data latih +
validasi + data uji) hanya jika tujuan berikutnya adalah membangun model
produksi untuk inferensi seluruh dataset.

Jika dilakukan retraining dengan seluruh data anotasi:

-   gunakan konfigurasi yang telah dipilih sebelumnya;
-   jangan melakukan tuning ulang berdasarkan hasil data uji;
-   catat bahwa model final untuk inferensi berbeda dari checkpoint yang
    dievaluasi pada data uji;
-   simpan checkpoint final dan prosedur retraining.

Untuk pendekatan leksikon, gunakan kamus dan aturan final yang telah
dibekukan. Tidak ada proses retraining classifier.

### 11.2 Prosedur inferensi

1.  Muat model atau kamus final.
2.  Baca seluruh ulasan yang belum diproses.
3.  Identifikasi kategori aspek yang perlu diprediksi.
4.  Bentuk pasangan ulasan-aspek.
5.  Terapkan preprocessing yang sesuai dengan pendekatan terpilih.
6.  Prediksi polaritas setiap pasangan.
7.  Simpan hasil prediksi beserta identitas ulasan dan aspek.
8.  Lakukan pemeriksaan kualitas hasil inferensi.

### 11.3 Format hasil inferensi

Simpan hasil dalam format terstruktur, misalnya:

  ----------------------------------------------------------------------------------------------
  review_id   review_text   aspect      predicted_sentiment     confidence/score model_version
  ----------- ------------- ----------- --------------------- ------------------ ---------------

  ----------------------------------------------------------------------------------------------

Kolom confidence/score bersifat opsional dan harus sesuai dengan
kemampuan model. Jangan menyamakan skor leksikon, decision function SVM,
dan probabilitas IndoBERT sebagai confidence yang langsung sebanding.

Untuk SVM, `SVC` tidak menghasilkan probabilitas secara default. Jika
probabilitas dibutuhkan, aktifkan mekanisme probabilitas atau gunakan
decision score dengan penjelasan yang tepat. Untuk IndoBERT,
probabilitas dapat diperoleh dari softmax logits, tetapi tetap bukan
jaminan kalibrasi.

------------------------------------------------------------------------

## 12. Struktur implementasi kode yang disarankan

Contoh struktur proyek:

``` text
absa-hotel/
├── data/
│   ├── raw/
│   │   └── annotated_reviews.csv
│   ├── processed/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   └── manifests/
│       └── split_manifest.json
├── configs/
│   ├── experiment.yaml
│   ├── lexicon.yaml
│   ├── svm.yaml
│   └── indobert.yaml
├── lexicons/
│   ├── inset.csv
│   ├── hotel_domain_lexicon.csv
│   └── aspect_terms.csv
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── split_data.py
│   │   └── validate_data.py
│   ├── preprocessing/
│   │   └── text_preprocessing.py
│   ├── models/
│   │   ├── lexicon_model.py
│   │   ├── svm_model.py
│   │   └── indobert_model.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── confusion_matrix.py
│   │   └── significance_test.py
│   └── inference/
│       └── predict_all.py
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_run_lexicon.py
│   ├── 03_train_svm.py
│   ├── 04_train_indobert.py
│   ├── 05_evaluate_test.py
│   └── 06_infer_full_dataset.py
├── outputs/
│   ├── predictions/
│   ├── metrics/
│   ├── figures/
│   ├── models/
│   └── logs/
├── requirements.txt
└── README.md
```

Struktur ini merupakan rekomendasi. Nama folder dan file dapat
disesuaikan, tetapi pemisahan data, model, evaluasi, dan artefak
sebaiknya dipertahankan.

### 12.1 Konfigurasi eksperimen

Simpan parameter utama dalam file konfigurasi agar eksperimen dapat
diulang tanpa mengubah kode, misalnya:

``` yaml
experiment:
  name: absa_hotel_model_comparison
  seed: 42
  split:
    train: 0.70
    validation: 0.15
    test: 0.15
    group_column: review_id

task:
  input_unit: review_aspect_pair
  text_column: review_text
  aspect_column: aspect
  label_column: sentiment
  labels:
    - negative
    - neutral
    - positive
  primary_metric: macro_f1

lexicon:
  base_lexicon: InSet
  domain_extension: lexicons/hotel_domain_lexicon.csv
  no_match_policy: neutral

svm:
  vectorizer:
    ngram_range:
      - [1, 1]
      - [1, 2]
  classifier:
    kernel:
      - linear
    C:
      - 0.1
      - 1
      - 10
    class_weight:
      - null
      - balanced

indobert:
  checkpoint: indobert-base-p2
  max_length:
    - 128
    - 256
  learning_rate:
    - 0.00002
    - 0.00003
    - 0.00005
  batch_size:
    - 8
    - 16
  epochs:
    - 2
    - 3
    - 4
  weight_decay: 0.01

evaluation:
  metrics:
    - accuracy
    - macro_precision
    - macro_recall
    - macro_f1
    - weighted_f1
  significance_test: mcnemar
  alpha: 0.05
  p_value_correction: holm
```

Konfigurasi tersebut adalah template awal. Sesuaikan nama checkpoint
dengan identifier resmi yang digunakan, jumlah label dengan codebook
anotasi, dan kandidat hyperparameter dengan ukuran dataset serta
kapasitas perangkat keras.

------------------------------------------------------------------------

## 13. Urutan eksekusi eksperimen

  -----------------------------------------------------------------------
  Tahap                   Aktivitas               Output yang harus
                                                  tersedia
  ----------------------- ----------------------- -----------------------
  1                       Validasi dataset dan    Dataset bersih dan
                          skema anotasi           laporan kualitas

  2                       Membentuk pasangan      Dataset ABSA
                          ulasan-aspek            

  3                       Split 70:15:15          Train, validation,
                          berdasarkan review_id   test, manifest

  4                       Menyusun InSet dan      Leksikon final dan
                          ekstensi domain         aturan

  5                       Menjalankan pendekatan  Prediksi validasi dan
                          leksikon                uji

  6                       Fit TF-IDF dan tuning   Pipeline SVM terbaik
                          SVM                     

  7                       Fine-tuning dan tuning  Checkpoint IndoBERT
                          IndoBERT                terbaik

  8                       Membekukan seluruh      Catatan konfigurasi
                          konfigurasi             final

  9                       Prediksi ketiga model   File prediksi gabungan
                          pada test yang sama     

  10                      Evaluasi keseluruhan    Tabel metrik dan
                          dan per aspek           confusion matrix

  11                      McNemar dan koreksi     Tabel p-value
                          Holm                    

  12                      Pemilihan model terbaik Keputusan model dan
                                                  justifikasi

  13                      Inferensi seluruh       Dataset dengan hasil
                          dataset                 prediksi ABSA

  14                      Penyusunan publikasi    Tabel, gambar, dan
                                                  analisis hasil
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 14. Checklist sebelum eksperimen dijalankan

### Dataset dan anotasi

-   [ ] Setiap baris memiliki `review_id`, teks ulasan, aspek, dan label
    sentimen.
-   [ ] Definisi label dan daftar aspek telah dibekukan.
-   [ ] Duplikasi dan kualitas anotasi telah diperiksa.
-   [ ] Semua pasangan dari satu `review_id` berada pada subset yang
    sama.
-   [ ] Proporsi split dan distribusi kelas telah diperiksa.
-   [ ] Data uji tidak digunakan untuk menyusun kamus atau tuning.

### Model

-   [ ] Leksikon InSet dan ekstensi domain memiliki versi yang jelas.
-   [ ] Aturan negasi, intensifier, dan fallback telah ditentukan.
-   [ ] TF-IDF hanya di-fit pada data latih.
-   [ ] SVM dituning menggunakan data validasi.
-   [ ] Checkpoint IndoBERT dan tokenizer telah dicatat.
-   [ ] IndoBERT dituning menggunakan data validasi.
-   [ ] Ketiga model menerima informasi aspek dan unit evaluasi yang
    sebanding.

### Evaluasi

-   [ ] Semua model memprediksi test set yang sama.
-   [ ] Tidak ada sampel uji yang dihilangkan tanpa alasan dan
    pelaporan.
-   [ ] Accuracy, precision, recall, dan F1-score dihitung.
-   [ ] Metrik macro dan per kelas dilaporkan.
-   [ ] Hasil per aspek dilaporkan.
-   [ ] Confusion matrix tersedia.
-   [ ] McNemar dilakukan untuk tiga pasangan model.
-   [ ] Koreksi Holm diterapkan pada tiga pengujian.
-   [ ] Keputusan model terbaik mempertimbangkan metrik dan signifikansi
    statistik.

### Reproducibility dan pelaporan

-   [ ] Seed, versi library, dan perangkat keras dicatat.
-   [ ] Konfigurasi dan checkpoint terbaik disimpan.
-   [ ] Prediksi per sampel disimpan.
-   [ ] Log training dan tuning tersedia.
-   [ ] Keterbatasan metode dan hasil dicatat.
-   [ ] Prosedur inferensi seluruh dataset dapat dijalankan ulang.

------------------------------------------------------------------------

## 15. Hal yang perlu dipastikan sebelum implementasi final

Beberapa detail tidak dapat ditentukan hanya dari flowchart dan
deskripsi penelitian. Tetapkan hal-hal berikut sebelum kode final
dibuat:

1.  **Skema anotasi:** apakah label sentimen terdiri dari
    positive/negative/neutral, atau terdapat label lain?
2.  **Representasi aspek:** apakah anotasi menyimpan kategori aspek
    saja, aspect term, atau keduanya?
3.  **Ekstraksi aspek:** apakah aspek untuk ketiga model sudah tersedia
    dari anotasi/daftar aspek, atau sistem harus menemukan aspek secara
    otomatis dari teks mentah?
4.  **Unit evaluasi:** apakah satu ulasan dapat memiliki lebih dari satu
    label untuk aspek yang sama?
5.  **Model konvensional:** apakah eksperimen utama hanya SVM, atau SVM
    dan Naive Bayes sama-sama wajib dibandingkan?
6.  **Ukuran dataset:** berapa jumlah ulasan unik dan pasangan
    ulasan-aspek? Informasi ini memengaruhi strategi split, batas
    pencarian hyperparameter, dan ukuran batch.
7.  **Checkpoint IndoBERT:** pastikan identifier resmi IndoBERT-base-p2
    yang akan digunakan.
8.  **Inferensi seluruh dataset:** apakah aspek yang akan diprediksi
    sudah ditentukan sebelumnya, atau perlu dilakukan aspect extraction
    otomatis?

**Rekomendasi prioritas:** pastikan poin 2--4 terlebih dahulu. Ketiganya
menentukan bentuk dataset, input model, dan validitas perbandingan.
Tanpa definisi unit ulasan-aspek yang konsisten, perbandingan antara
pendekatan leksikon, SVM, dan IndoBERT berisiko tidak mengukur tugas
yang sama.

------------------------------------------------------------------------

## 16. Prinsip metodologis yang tidak boleh dilanggar

1.  Ketiga pendekatan harus dievaluasi pada unit ulasan-aspek dan data
    uji yang identik.
2.  Pemisahan data dilakukan berdasarkan ulasan unik untuk mencegah
    kebocoran antar-aspek dari ulasan yang sama.
3.  Data uji hanya digunakan untuk evaluasi final, bukan untuk tuning.
4.  TF-IDF hanya mempelajari vocabulary dan IDF dari data latih.
5.  Ekstensi leksikon tidak boleh disusun dengan memanfaatkan label atau
    kesalahan pada data uji.
6.  IndoBERT menggunakan tokenizer bawaan dan tidak menerima
    preprocessing agresif seperti stemming atau stopword removal.
7.  Semua keputusan konfigurasi harus dicatat agar eksperimen dapat
    direproduksi.
8.  McNemar's test dilakukan pada prediksi berpasangan per unit
    evaluasi, bukan pada skor agregat.
9.  Perbedaan yang signifikan secara statistik tidak otomatis berarti
    perbedaan yang besar secara praktis.
10. Model yang dipilih harus memiliki justifikasi berbasis performa,
    stabilitas, dan kebutuhan implementasi.

Dokumen ini menjadi spesifikasi eksperimen utama. Setiap perubahan
terhadap unit analisis, skema label, split, preprocessing, atau protokol
evaluasi harus dicatat sebagai perubahan metodologi, bukan sekadar
perubahan implementasi.
