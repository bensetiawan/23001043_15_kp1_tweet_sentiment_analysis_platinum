# Data Science Project - Tweet Sentiment Analysis

![_3d8508af-d3f5-44e2-88bd-d37d5d450ecd](https://github.com/bensetiawan/2300968_Sentiment_Analysis_Platinum/assets/93572380/52981cd0-9c00-472e-8d1d-0e625a026910)

## Background

Media sosial, khususnya Twitter atau X, telah menjadi platform populer untuk mengekspresikan opini dan sentimen publik terhadap berbagai topik. Analisis sentimen tweet dapat membantu memahami persepsi publik terhadap suatu produk, layanan, atau peristiwa. Laporan ini akan membahas bagaimana Machine Learning dapat digunakan untuk menganalisis sentimen tweet.

## Business Understanding

Tujuan dari analisis sentimen tweet adalah untuk mendapatkan wawasan tentang persepsi publik terhadap suatu topik. Wawasan ini dapat membantu suatu perusahaan atau instusi untuk:

- Meningkatkan reputasi dan citra merek
- Mengembangkan produk dan layanan yang lebih baik
- Memantau efektivitas kampanye pemasaran
- Mengidentifikasi krisis dan isu yang berpotensi

## Problem Statements

Analisis sentimen tweet secara manual adalah proses yang memakan waktu dan sulit. Machine Learning dapat membantu mengotomatisasi proses ini dan meningkatkan skalabilitas.

## Goals

Tujuan dari proyek ini adalah untuk membangun model Machine Learning yang dapat secara otomatis mengklasifikasikan sentimen text atau tweet (positif, negatif, atau netral).

**Solution statements**
Membangun 2 model machine learning yaitu LSTM, dan MLP :
- **LSTM** adalah jenis jaringan saraf tiruan  yang dirancang untuk mempelajari ketergantungan jangka panjang dalam data, sehingga sangat cocok untuk tugas-tugas seperti: Pengenalan suara, penerjemahaan bahasa, dan analisis sentiment.
- **MLP** adalah jenis jaringan saraf tiruan yang terdiri dari beberapa lapisan neuron yang saling terhubung satu sama lain.MLP digunakan untuk mempelajari pola-pola yang kompleks dan non-linear pada data input seperti, pengenalan wajah, prediksi harga saham, dan pengenalan pola pada teks.

## Data Understanding
Data yang digunakan untuk proyek ini adalah dataset tweet yang berisi tweet orang - orang indonesia tentang topik tertentu. Dataset ini harus dilakukan cleansing terlebih dahulu dan preprocessing sebelum digunakan untuk pelatihan model. Dalam data tersebut terdapat 3 Sentiment, yaitu :
- Negative
- Neutral
- Positive

## Data Preparation
Langkah-langkah berikut dilakukan untuk menyiapkan data sebelum dilatih kedalam model :

 - Mengubah teks menjadi lowercase
 - Menghapus noise seperti hashtag, retweet, mention, emoji, link, dan special character
 - Mengubah Kata Alay menjadi kata Tidak Alay 
 - Tokenisasi
 - Remove Stopwords
 - Melakukan Stemming
 - Tf-Idf (MLP)
 - Sequences dan Padding (LSTM)

## Modeling
Terdapat 2 model yang dibangun yaitu, Model menggunakan konsep LSTM dan model dengan konsep MLP

**LSTM (Long Short-Term Memory)**: 
- Kelebihan LSTM :
    - Memori jangka panjang
    - Kemampuan belajar dependensi jarak jauh
    - Akurasi tinggi
- optimizer yang diterapkan adalah Adam dengan learning rate (0.0001)

 **MLP (Multi Layer-Perceptron)**:
- Kelebihan MLP :
    - Mudah dipahami dan diimplementasikan
    - Kemampuan Generalisasi yang baik
    - Fleksibel
- MLP yang dipakai adalah **MLPClassifier**, dengan activation = 'relu', solver = 'adam', hidden_layer = 30, alpha = '0.008', dan epsilon = 1e-5

## Evaluation
Metrics evaluasi yang dipakai adalah **Classification Report**. Didalam Classification Report terdapat beberapa metrics yaitu, **akurasi, precision, recall, dan F1 score**.

Berdasarkan hasil evaluasi, didapatkan model LSTM memiliki performa yang lebih baik dibandingkan MLP. Dengan rata - rata akurasi setelah dilakukan cross validation adalah **0.91** sedangkan model MLP adalah **0.8245**

## API
Model - Model tersebut diintegrasikan kedalam API. API dibangun menggunakan API Flask, dan Swagger UI. API yang dibangun dapat digunakan untuk Analisa Sentiment dalam bentuk text dan file yang berisikan data - data Tweets.

![lstm model - Copy](https://github.com/bensetiawan/2300968_Sentiment_Analysis_Platinum/assets/93572380/76f0581d-ba35-4154-aac9-5b18678a421f)
