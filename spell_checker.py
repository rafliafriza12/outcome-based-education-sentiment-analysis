import re
import collections

# Fungsi untuk membaca corpus dan membuat dictionary frekuensi kata
def words(text):
    return re.findall(r'\w+', text.lower())

# Bangun model dari dataset bahasa Inggris
with open("big.txt", "r", encoding="utf-8") as f:
    WORDS = collections.Counter(words(f.read()))

# Fungsi untuk menghitung probabilitas kata
def P(word, N=sum(WORDS.values())):
    return WORDS[word] / N

# Menghasilkan daftar kata yang berjarak satu edit dari kata input
def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

# Menghasilkan daftar kata dengan edit distance hingga 2
def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# Mencari kata yang paling mungkin berdasarkan kemungkinan koreksi
def known(words):
    return set(w for w in words if w in WORDS)

def correction(word):
    return max(known([word]) or known(edits1(word)) or known(edits2(word)) or [word], key=P)

# Contoh penggunaan
print(correction("speling"))  # Output: "spelling"
print(correction("korreksi"))  # Output: kemungkinan kata yang paling mirip