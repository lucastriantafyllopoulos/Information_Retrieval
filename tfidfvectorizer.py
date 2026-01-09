import matplotlib.pyplot as plt
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

RESULTS_FILE = "results.txt" #Το αρχείο με τα αποτελέσματα από το 2ο ερώτημα
RELEVANT_FILE = "Relevant.txt" #Το αρχείο με τα σχετιζόμενα κείμενα για κάθε query
OUTPUT_FOLDER = "question3_diagrams"  #Ο φάκελος που αποθηκεύονται όλες οι εικόνες με τις καμπύλες ακρίβειας-ανάκλησης
QUERIES_FILE = "Queries.txt" #Το αρχείο με τα 20 ερωτήματα
DOCS_FOLDER = "docs" #Ο φάκελος με έγγραφα

def normalize_id(raw):
    #Αναγνώριση των αρχείων από τον φάκελο "docs" με αφαίρεση
    #των μηδενικών από το μπροστά μέρος, καθώς τα id των κειμένων
    #στο Relevant.txt είναι ακέραιοι

    if raw is None:
        return raw
    s = str(raw).strip()
    try:
        return str(int(s))
    except Exception:

        #Αφαιρεί leading zeros γιατί τα 
        #αρχεία ξεκινούν με 0 (ένα ή πολλά)
        s2 = s.lstrip("0")
        return s2 if s2 != "" else s #αν έγινε άδειο, επιστρέφει το αρχικό

def load_relevant_docs(filepath=RELEVANT_FILE):
    #Φόρτωση του αρχείου με τα σχετιζόμενα για το εκάστοτε ερώτημα κείμενα
    relevant_list = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                relevant_list.append(set())
                continue
            ids = line.split()
            relevant_list.append({normalize_id(x) for x in ids})
    return relevant_list

def load_results(filepath=RESULTS_FILE):
    #Φόρτωση του αρχείου των αποτελεσμάτων του 2ου ερωτήματος
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    current = None
    mode = None #"tfidf" ή "prob"

    while i < len(lines):
        line = lines[i].strip()

        if line == "":
            i += 1
            continue

        if line.startswith("Query"):
            #Νέο block query
            current = {"tfidf": [], "prob": []}
            results.append(current)
            i += 1
            continue

        #Φόρτωση αποτελεσμάτων τύπου TF-IDF
        if line.startswith("Top TF-IDF"):
            mode = "tfidf"
            i += 1
            continue
        #Φόρτωση αποτελεσμάτων τύπου Πιθανοτικού (Probabilistic)
        if line.startswith("Top Probabilistic"):
            mode = "prob"
            i += 1
            continue

        if mode in ("tfidf", "prob"):
            parts = line.split()
            doc_id = normalize_id(parts[0])
            current[mode].append(doc_id)

        i += 1

    return results

#Μετρική Ακρίβειας
def precision(retrieved, relevant_set):
    if not retrieved:
        return 0.0
    tp = sum(1 for d in retrieved if d in relevant_set)
    #relevant_set = αριθμός εγγράφων που ανακτήθηκαν
    #retrieved = αριθμός εγγράφων που ανακτήθηκαν
    return tp / len(retrieved)

#Μετρική Ανάκλησης
def recall(retrieved, relevant_set):
    #relevant_set = Αριθμός σχετικών κειμένων στη συλλογή
    if not relevant_set:
        return 0.0
    tp = sum(1 for d in retrieved if d in relevant_set)
        #retrieved = Αριθμός σχετικών ανακτηθέντων κειμένων
    return tp / len(relevant_set)

#Μετρική F1
def f1_score(p, r):
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def precision_at_k(retrieved, relevant_set, k):
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    tp = sum(1 for d in retrieved_k if d in relevant_set)
    return tp / k

#Η καμπύλη ενημερώνεται μόνο όταν ανακτάται ένα σχετικό έγγραφο
#έτσι κάθε σημείο της καμπύλης αντιστοιχεί σε πραγματική αύξηση της ανάκλησης
def precision_recall_hits(ranked_docs, relevant_set):
    R = len(relevant_set)
    if R == 0:
        return [0.0], [0.0]

    hits = 0
    #Στο recall=0 δεν έχει ανακτηθεί κανένα έγγραφο, άρα δεν υπάρχει σφάλμα·
    #το σημείο χρησιμοποιείται μόνο ως σημείο εκκίνησης της καμπύλης.
    #Προσοχή!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    recalls = []
    precisions = []

    for i, d in enumerate(ranked_docs, start=1):
        if d in relevant_set:
            hits += 1
            recalls.append(hits / R)
            precisions.append(hits / i)

    return recalls, precisions

def load_documents():
    doc_ids = []
    texts = []

    for fname in sorted(os.listdir(DOCS_FOLDER)):
        full_path = os.path.join(DOCS_FOLDER, fname)

        # Αγνόησε φακέλους
        if not os.path.isfile(full_path):
            continue

        doc_id = normalize_id(fname)

        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

            # Ασφάλεια: αγνόησε τελείως άδεια αρχεία
            if text == "":
                continue

            texts.append(text)
            doc_ids.append(doc_id)

    return doc_ids, texts



def load_queries():
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        return [q.strip() for q in f]


# ======== Plot ========

def plot_pr_hits(r_tf, p_tf, r_pr, p_pr, r_vec, p_vec, qid):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.plot(r_tf, p_tf, marker="o", label="TF-IDF (χειροποίητο)")
    plt.plot(r_pr, p_pr, marker="o", label="Πιθανοτική")
    plt.plot(r_vec, p_vec, marker="o", label="TF-IDF (Vectorizer)")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall (Hits-only) – Query {qid}")
    plt.grid(True)
    plt.legend()

    outpath = os.path.join(OUTPUT_FOLDER, f"question3_query{qid}.png")
    plt.savefig(outpath, dpi=150)
    plt.close()


# ======== Main ========

def main():

    # ===============================
    # Φόρτωση δεδομένων
    # ===============================
    relevant_list = load_relevant_docs(RELEVANT_FILE)
    results = load_results(RESULTS_FILE)

    # Φόρτωση συλλογής εγγράφων (ΟΧΙ από φάκελο τυχαία,
    # αλλά με την ίδια λογική που δουλεύει το project)
    doc_ids, documents = load_documents()
    doc_id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    # Φόρτωση ερωτημάτων
    queries = load_queries()

    n = min(len(relevant_list), len(results), len(queries))
    K_VALUES = [5, 10, 50, 100, 200]

    print(f"Επεξεργασία {n} ερωτημάτων.")

    # ===============================
    # TF-IDF Vectorizer (Βιβλιοθήκη)
    # ===============================
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=1,
        max_df=0.9,
        norm="l2"
    )

    t0 = time.time()
    doc_matrix = vectorizer.fit_transform(documents)
    vectorizer_time = time.time() - t0

    print(f"\nTF-IDF Vectorizer fitted σε {vectorizer_time:.4f} sec")

    # ===============================
    # Ανά query
    # ===============================
    for qidx in range(n):
        qid = qidx + 1
        relevant_set = relevant_list[qidx]

        tfidf_ranked = results[qidx]["tfidf"]
        prob_ranked = results[qidx]["prob"]

        # -------------------------------
        # TF-IDF Vectorizer ranking
        # -------------------------------
        t0 = time.time()
        q_vec = vectorizer.transform([queries[qidx]])
        scores = (doc_matrix @ q_vec.T).toarray().ravel()
        vec_time = time.time() - t0

        ranked_indices = np.argsort(scores)[::-1]
        vec_ranked = [
            doc_ids[i] for i in ranked_indices if scores[i] > 0
        ]

        # -------------------------------
        # Μετρικές
        # -------------------------------
        p_tf = precision(tfidf_ranked, relevant_set)
        r_tf = recall(tfidf_ranked, relevant_set)
        f1_tf = f1_score(p_tf, r_tf)

        p_pr = precision(prob_ranked, relevant_set)
        r_pr = recall(prob_ranked, relevant_set)
        f1_pr = f1_score(p_pr, r_pr)

        p_vec = precision(vec_ranked, relevant_set)
        r_vec = recall(vec_ranked, relevant_set)
        f1_vec = f1_score(p_vec, r_vec)

        # -------------------------------
        # Εκτύπωση
        # -------------------------------
        print("\n" + "=" * 60)
        print(f"Ερώτημα {qid} | Σχετικά έγγραφα: {len(relevant_set)}")

        print(f"TF-IDF (χειροκίνητο) ==> P={p_tf:.4f}, R={r_tf:.4f}, F1={f1_tf:.4f}")
        print(f"Probabilistic       ==> P={p_pr:.4f}, R={r_pr:.4f}, F1={f1_pr:.4f}")
        print(f"TF-IDF Vectorizer   ==> P={p_vec:.4f}, R={r_vec:.4f}, F1={f1_vec:.4f}")
        print(f"Χρόνος Vectorizer query: {vec_time:.4f} sec")

        for k in K_VALUES:
            print(
                f"P@{k}: "
                f"TF-IDF={precision_at_k(tfidf_ranked, relevant_set, k):.4f} | "
                f"Prob={precision_at_k(prob_ranked, relevant_set, k):.4f} | "
                f"Vec={precision_at_k(vec_ranked, relevant_set, k):.4f}"
            )

        # -------------------------------
        # PR hits-only καμπύλες
        # -------------------------------
        r_tf_h, p_tf_h = precision_recall_hits(tfidf_ranked, relevant_set)
        r_pr_h, p_pr_h = precision_recall_hits(prob_ranked, relevant_set)
        r_vec_h, p_vec_h = precision_recall_hits(vec_ranked, relevant_set)

        # -------------------------------
        # Διάγραμμα (3 γραμμές)
        # -------------------------------
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        plt.figure(figsize=(7, 6))
        plt.plot(r_tf_h, p_tf_h, marker="o", label="TF-IDF (manual)")
        plt.plot(r_pr_h, p_pr_h, marker="o", label="Probabilistic")
        plt.plot(r_vec_h, p_vec_h, marker="o", label="TF-IDF Vectorizer")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision–Recall (Hits-only) – Query {qid}")
        plt.grid(True)
        plt.legend()

        outpath = os.path.join(OUTPUT_FOLDER, f"question3_query{qid}.png")
        plt.savefig(outpath, dpi=150)
        plt.close()

    print("\nΤέλος!")



if __name__ == "__main__":
    main()
