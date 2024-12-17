import re
from collections import defaultdict
import math

# Preprocessing Teks
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

# Training Naive Bayes
def train_naive_bayes(data):
    category_word_counts = defaultdict(lambda: defaultdict(int))
    category_counts = defaultdict(int)
    vocabulary = set()

    for category, text in data:
        words = preprocess(text)
        category_counts[category] += 1
        for word in words:
            category_word_counts[category][word] += 1
            vocabulary.add(word)
    return category_word_counts, category_counts, vocabulary

# Prediksi Kategori
def predict_naive_bayes(text, category_word_counts, category_counts, vocabulary, total_docs):
    words = preprocess(text)
    scores = {}
    vocab_size = len(vocabulary)

    for category in category_counts:
        log_prob = math.log(category_counts[category] / total_docs)
        total_words = sum(category_word_counts[category].values())

        for word in words:
            word_count = category_word_counts[category][word] + 1
            log_prob += math.log(word_count / (total_words + vocab_size))
        
        scores[category] = log_prob
    
    return max(scores, key=scores.get), scores

# Data Training
data = [
    ("Politik", "Presiden memberikan pidato ekonomi"),
    ("Politik", "Menteri luar negeri melakukan kunjungan ke negara tetangga"),
    ("Politik", "Parlemen menyetujui undang-undang baru tentang pendidikan"),
    ("Politik", "Debat calon presiden berlangsung sengit semalam"),
    ("Politik", "Demonstrasi besar-besaran menuntut kebijakan pemerintah"),
    ("Politik", "Kebijakan ekonomi baru dipuji oleh para ahli politik"),
    ("Politik", "Presiden mengunjungi daerah yang terkena bencana alam"),
    ("Olahraga", "Tim nasional memenangkan pertandingan sepak bola"),
    ("Olahraga", "Pebalap muda meraih podium pertama di ajang internasional"),
    ("Olahraga", "Pemain bulu tangkis Indonesia berhasil meraih medali emas"),
    ("Olahraga", "Atlet renang mencetak rekor baru dalam kompetisi nasional"),
    ("Olahraga", "Klub sepak bola terkenal merekrut pemain bintang dari luar negeri"),
    ("Olahraga", "Kejuaraan dunia atletik diadakan di stadion utama"),
    ("Olahraga", "Petinju Indonesia menang dalam pertarungan kelas dunia"),
    ("Teknologi", "Perusahaan meluncurkan produk smartphone baru"),
    ("Teknologi", "Penemuan teknologi AI semakin mempermudah kehidupan manusia"),
    ("Teknologi", "Startup teknologi lokal mendapat pendanaan dari investor asing"),
    ("Teknologi", "Teknologi 5G mulai diterapkan di berbagai kota besar"),
    ("Teknologi", "Perusahaan merilis sistem operasi terbaru untuk perangkat komputer"),
    ("Teknologi", "Inovasi robotika membawa perubahan besar dalam industri manufaktur"),
    ("Teknologi", "Keamanan siber menjadi tantangan utama di era digital"),
    ("Bisnis", "Pasar saham naik signifikan hari ini"),
    ("Bisnis", "Perusahaan logistik mencatat keuntungan besar pada kuartal ini"),
    ("Bisnis", "Investasi di bidang energi terbarukan semakin meningkat"),
    ("Bisnis", "Perekonomian global dipengaruhi oleh krisis di beberapa negara"),
    ("Bisnis", "Pelaku usaha kecil dan menengah mendapatkan insentif dari pemerintah"),
    ("Bisnis", "Ekspor komoditas utama mengalami peningkatan yang signifikan"),
    ("Bisnis", "Konsumen mulai beralih ke produk ramah lingkungan dalam skala besar"),
    ("Bisnis", "Harga mobil turun jalanan jadi macet"),
]

# Training
category_word_counts, category_counts, vocabulary = train_naive_bayes(data)
total_docs = sum(category_counts.values())

# Inputan Keyboard untuk Teks Baru
new_text = input("Masukkan teks berita baru: ")
predicted_category, scores = predict_naive_bayes(new_text, category_word_counts, category_counts, vocabulary, total_docs)

# Hasil
print("Teks Baru:", new_text)
print("Prediksi Kategori:", predicted_category)
print("Skor Probabilitas:")
for category, score in scores.items():
    print(f"  {category}: {score:.4f}")
