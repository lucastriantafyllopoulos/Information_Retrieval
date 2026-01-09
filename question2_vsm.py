import os
import math
import pickle
import time
from collections import defaultdict, Counter
from pathlib import Path
import re

#path = Path("docs")
#num_files = sum(1 for p in path.iterdir() if p.is_file())

INVERTED_INDEX_FILE = "inverted_index.pkl" #Το ανεστραμμένο ευρετήριο
QUERIES_FILE = "Queries.txt" #Το αρχείο με τα ερωτήματα
#TOP_RESULTS_AMOUNT = num_files #Το πλήθος των πιο σχετικών αποτελεσμάτων που θέλουμε αν εμφανιστεί

#Διαχωρισμός των λέξεων ως ξεχωριστά tokens
def tokenize_text(text):
    #return text.lower().split()
    return re.findall(r"[a-z0-9]+", text.lower())

#Φόρτωση του ανεστραμμένου αρχείου
def load_inverted_index(filename):
    with open(filename, "rb") as f:
        inverted_index = pickle.load(f)
    return inverted_index

#Εξαγωγή συνόλου υπαρχόντων εγγράφων από το inverted index
def get_existing_docs_from_index(inverted_index):
    docs = set()
    for postings in inverted_index.values():
        for doc_id, tf in postings:
            docs.add(doc_id)
    return sorted(docs)

#Υπολογισμός του IDF για κάθε όρο
def calculate_IDF(inverted_index, N):
    #Το N είναι ο συνολικός αριθμός εγγράφων στην συλλογή μετά τη διαχείριση των απόντων αρχείων
    #Το n είναι ο αριθμός εγγράφων στα οποία αντισχτοιεί ένας όρος
    idf = {}
    for term, postings in inverted_index.items():
        n = len(postings)
        if n == 0:
            idf[term] = 0.0
        else:
            #υπολογισμός του τύπου log(N/n), προστασία από n==N
            #χρήση της βάσης του 10 για τον λογάριθμο
            idf[term] = math.log10(N / n) if n < N else 0.0
    return idf

#Υπολογισμός βαρών TF-IDF
def calculate_weights_TF_IDF(term_freqs, idf):
    tf_idf = {}
    for term, tf in term_freqs.items():
        tf_idf[term] = tf * idf.get(term, 0)
    return tf_idf

#Υπολογισμός Βαρών Όρων Ερωτήματος (Query term weight) για καλύτερο πιθανοτικό με βάση το paper του Shalton
def calculate_query_prob_weights(query_terms, N, inverted_index):
    weights = {}
    for term in query_terms:
        n = len(inverted_index.get(term, []))
        if n == 0 or n == N:
            #Λαμβάνουμε υπόψη την περίπτωση να έχουμε n==0 ή n==N
            weights[term] = 0.0
        else:
            #Query term weight για καλύτερο πιθανοτικό
            weights[term] = math.log10((N - n) / n)
    return weights

#Υπολογισμός Βαρών Εγγράφων Ερωτήματος (Query term weight) για καλύτερο πιθανοτικό με βάση το paper του Shalton
def calculate_probabilistic_doc_vectors(inverted_index):

    #Πρώτα φτιάχνουμε tf ανά έγγραφο και βρίσκουμε max_tf ανά έγγραφο
    doc_term_tfs = defaultdict(lambda: defaultdict(int))
    max_tf_per_doc = defaultdict(int)

    for term, postings in inverted_index.items():
        for doc_id, tf in postings:
            doc_term_tfs[doc_id][term] = tf
            if tf > max_tf_per_doc[doc_id]:
                max_tf_per_doc[doc_id] = tf

    #Έπειτα φτιάχνουμε τα βάρη
    prob_doc_vectors = defaultdict(dict)
    for doc_id, term_tf_map in doc_term_tfs.items():
        max_tf = max_tf_per_doc.get(doc_id, 1) or 1
        for term, tf in term_tf_map.items():
            prob_doc_vectors[doc_id][term] = 0.5 + 0.5 * (tf / max_tf)
    return prob_doc_vectors


#Υπολογισμός του συνιμητόνου της γωνίας (μεταξύ δύο διανυσμάτων) 
#για την εύρεση ομοιότητας μεταξύ ερωτήματος και εγγράφων
def cosine_similarity(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[t] * vec2[t] for t in common)
    denom1 = math.sqrt(sum(v**2 for v in vec1.values()))
    denom2 = math.sqrt(sum(v**2 for v in vec2.values()))
    if denom1 == 0 or denom2 == 0:
        return 0
    return numerator / (denom1 * denom2)

#Υπολογισμός TF-IDF διανυσμάτων για όλα τα έγγραφα
def calculate_tfidf_doc_vectors(inverted_index, idf):
    doc_vectors = defaultdict(dict)
    for term, postings in inverted_index.items():
        for doc_id, tf in postings:
            doc_vectors[doc_id][term] = tf * idf.get(term, 0)
    return doc_vectors

#Κύριο Πρόγραμμα
if __name__ == "__main__":

    #Φόρτωση του ανεστραμμένου ευρετηρίου
    inverted_index = load_inverted_index(INVERTED_INDEX_FILE)

    #Υπολογισμός του συνόλου των υπαρχόντων εγγράφων από το ίδιο το ανεστραμμένο ευρετήριο
    existing_docs = get_existing_docs_from_index(inverted_index)

    total_docs = len(existing_docs)
    print(f"Το ανεστραμμένο ευρετήριο φορτώθηκε! #documents (existing) = {total_docs}\n")

    #Φόρτωση όλων των εγγράφων. Το Ν είναι βασισμένο από τα συνολικά υπάρχοντα αρχεία
    idf = calculate_IDF(inverted_index, total_docs)

    #Αρχεία TF-IDF διανυσμάτων
    doc_vectors = calculate_tfidf_doc_vectors(inverted_index, idf)

    #Αρχεία Πιθανοτικής τεχνικής
    prob_doc_vectors = calculate_probabilistic_doc_vectors(inverted_index)

    #Ανάγνωση των ερωτημάτων από το Queries.txt ώστε να εμφανίζονται οργανωμένα τα αποτελέσματα
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    #Ανοίγουμε το αρχείο
    out = open("results.txt", "w", encoding="utf-8")

    #Επεξεργασία κάθε ερωτήματος
    for query_id, query_text in enumerate(queries, 1):
        print(f"Ερώτημα {query_id}: {query_text}")

        tokens = tokenize_text(query_text)
        query_tf = Counter(tokens)

        #Υπολογισμός βαρών για TF-IDF και Καλύτερου Πιθανοτικού για κάθε ερώτημα
        query_tfidf = calculate_weights_TF_IDF(query_tf, idf)

        # Σημειώση: το weight του query είναι probabilistic idf (log((N-n)/n)), χωρίς tf
        query_prob = calculate_query_prob_weights(query_tf.keys(), total_docs, inverted_index)
        
        #Υπολογισμός ομοιότητας με κάθε έγγραφο
        #Χρονομέτρηση TF-IDF για κάθε ερώτημα
        start_tfidf = time.time()
        tf_idf_scores = {}
        
        for doc_id, doc_vector in doc_vectors.items():
            tf_idf_scores[doc_id] = cosine_similarity(query_tfidf, doc_vector)
        
        #Ταξινόμηση των πιο σχετικών αποτελεσμάτων TF-IDF κατά φθίνουσα σειρά
        top_tf_idf = sorted(tf_idf_scores.items(), key=lambda x: x[1], reverse=True)[:total_docs]
        end_tfidf = time.time()
        tfidf_time = end_tfidf - start_tfidf   

        #Χρονομέτρηση Πιθανοτικής τεχνικής για κάθε ερώτημα
        start_prob = time.time()
        prob_scores = {}
        
        #for doc_id, doc_vector in doc_vectors.items():
        for doc_id, doc_vector in prob_doc_vectors.items():
            prob_scores[doc_id] = cosine_similarity(query_prob, doc_vector)

        #Ταξινόμηση των πιο σχετικών αποτελεσμάτων Πιθανοτικής τεχνικής σε φθίνουσα σειρά
        top_prob = sorted(prob_scores.items(), key=lambda x: x[1], reverse=True)[:total_docs]
        end_prob = time.time()
        prob_time = end_prob - start_prob


        #Εμφάνιση των πιο σχετικών αποτελεσμάτων από τα δύο είδη τεχνικών για κάθε ερώτημα
        print("Top TF-IDF:")
        print(f"Χρόνος TF-IDF: {tfidf_time:.6f} sec\n")
        for doc, score in top_tf_idf:
            print(f"Έγγραφο {doc} -> {score:.4f}")

        print("\nTop Probabilistic:")
        print(f"Χρόνος Πιθανοτικής τεχνικής: {prob_time:.6f} sec\n")
        for doc, score in top_prob:
            print(f"Έγγραφο {doc} -> {score:.4f}")

        print("\n")

        #Αποθήκευση των αποτελεσμάτων σε εξωτερικό αρχείο για 
        #να χρησιμοποιηθούν στο 3ο ερώτημα
        out.write(f"Query {query_id}: {query_text}\n\n")
        out.write("Top TF-IDF:\n\n")
        for doc, score in top_tf_idf:
            out.write(f"{doc} {score:.4f}\n")
        out.write("\n")
        out.write("Top Probabilistic:\n\n")
        for doc, score in top_prob:
            out.write(f"{doc} {score:.4f}\n")
        out.write("\n")

    out.close()
    print("Τα αποτελέσματα αποθηκεύτηκαν στο results.txt!")
