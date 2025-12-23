import os
import string
import pickle
from collections import defaultdict, Counter

DOCUMENTS_FOLDER = "docs" #Ο φάκελος που περιέχει τα κείμενα
DOCUMENTS_NUMBER = 1239  #Συνολικός αριθμός των αρχείων
OUTPUT_INVERTED_INDEX = "inverted_index.pkl" #Παραγόμενο αρχείο που αποθηκεύτεται το ανεστραμμένο ευρετήριο

#Διαχωρισμός των λέξεων ως ξεχωριστά tokens
def tokenize_text(text):
    return text.split()

#Δημιουργία του ανεστραμμένου ευρετηρίου
def create_inverted_index():
    inverted_index = defaultdict(list)
    existing_doc_ids = [] #Λίστα με αρχεία τύπου "File" που υπάρχουν στον φάκελο "docs"
    missing_docs = [] #Λίστα με αρχεία που δεν υπάρχουν στον φάκελο "docs" ακολουθώντας αριθμητική σειρά από 00001 έως 01239

    #Αναμένεται να υπάρχουν αρχεία 00001 έως 01239
    for i in range(1, DOCUMENTS_NUMBER + 1):
        filename = f"{i:05d}"  #Τα αρχεία έχουν ονόματα 5-ψήφιου αριθμού
        filepath = os.path.join(DOCUMENTS_FOLDER, filename)

        if not os.path.exists(filepath):
            missing_docs.append(filename) #Προσθήκη των id των αρχείων στην λίστα με τα απόντα αρχεία
            continue #Εαν λείπει κάποιο από τα αρχεία, απλά συνέχισε

        existing_doc_ids.append(filename) #Προσθήκη των id των αρχείων στην λίστα με τα υπάρχοντα αρχεία

        #Διάβασμα του περιεχομένου του κάθε αρχείου
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        #Επεξεργασία και μέτρηση συχνοτήτων
        tokens = tokenize_text(text)
        term_freqs = Counter(tokens)

        #Ενημέρωση ευρετηρίου
        for term, tf in term_freqs.items():
            inverted_index[term].append((filename, tf))

    print(f"Βρέθηκαν {len(existing_doc_ids)} αρχεία από {DOCUMENTS_NUMBER}")
    if missing_docs:
        print(f"Λείπουν {len(missing_docs)} αρχεία:")
        print(missing_docs)

    return inverted_index, existing_doc_ids, missing_docs

#Αποθήκευση του ανεστραμμένου ευρετηρίου σε εξωτερικό αρχείο σε δυαδική μορφή
def save_inverted_index(index, filename=OUTPUT_INVERTED_INDEX):

    with open(filename, "wb") as f:
        pickle.dump(index, f)
    print(f"Το ανεστραμμένο ευρετήριο αποθηκεύτηκε στο αρχείο '{filename}'!")

#Κύριο Πρόγραμμα
if __name__ == "__main__":
    inverted_index, existing_docs, missing_docs = create_inverted_index()
    save_inverted_index(inverted_index)
    #Το αρχείο περιέχει τους όρους, τα έγγραφα και το πόσο συχνά εμφανίζεται
    #ο εκάστοτε όρος, αλλά και μια λίστα με αρχεία που δεν υπάρχουν και ενδεχομενως
    #να ζητηθούν μετά σε επόμενα ερωτήματα