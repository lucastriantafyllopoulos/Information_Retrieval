import matplotlib.pyplot as plt
import os

RESULTS_FILE = "results.txt" #Το αρχείο με τα αποτελέσματα από το 2ο ερώτημα
RELEVANT_FILE = "Relevant.txt" #Το αρχείο με τα σχετιζόμενα κείμενα για κάθε query
OUTPUT_FOLDER = "question3_diagrams"  #Ο φάκελος που αποθηκεύονται όλες οι εικόνες με τις καμπύλες ακρίβειας-ανάκλησης

def normalize_id(raw):
    #Αναγνώριση των αρχείων από τον φάκελο "docs" με αφαίρεση
    #των μηδενικών από το μπροστά μέρος, καθώς τα id των κειμένων
    #στο Relevant.txt είναι ακέραιοι και 

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
    if k <= 0:
        return 0.0
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

#Δημιουργία και αποθήκευση διαγράμματος Ακρίβειας–Ανάκλησης
#με χρήση της μεθόδου precision_recall_hits_only  για δύο διαφορετικά μοντέλα ανάκτησης.
def plots_precision_recall_hits(recalls_tf, precisions_tf, recalls_pr, precisions_pr, qid):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.plot(recalls_tf, precisions_tf, marker="o", label="TF-IDF")
    plt.plot(recalls_pr, precisions_pr, marker="o", label="Πιθανοκρατική")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall (Hits-only) – Query {qid}")
    plt.grid(True)
    plt.legend()

    #Δημιουργία εικόνων διαγραμμάτων για κάθε query
    outpath = os.path.join(OUTPUT_FOLDER, f"question3_query{qid}.png")
    plt.savefig(outpath, dpi=150)
    plt.close()

    return outpath

def main():
    #Φόρτωμα του αρχείου με τα πραγματικά σχετικά κείμενα
    #και των αποτελεσμάτων από το 2ο ερώτημα
    relevant_list = load_relevant_docs(RELEVANT_FILE)
    results = load_results(RESULTS_FILE)

    n = min(len(relevant_list), len(results))
    K_VALUES = [5, 10, 50, 100, 200]

    print(f"Επεξεργασία {n} ερωτημάτων.")

    #Βρόγχος για εκκίνηση υπολογισμού των μετρικών και δημιουργία
    #διαγραμμάτων για τα 20 ερωτήματα
    for qidx in range(n):
        qid = qidx + 1
        relevant_set = relevant_list[qidx]
        tfidf_ranked = results[qidx]["tfidf"]
        prob_ranked = results[qidx]["prob"]

        #Υπολογισμός Μετρικών TF-IDF
        p_tf = precision(tfidf_ranked, relevant_set)
        r_tf = recall(tfidf_ranked, relevant_set)
        f1_tf = f1_score(p_tf, r_tf)

        #Υπολογισμός Μετρικών Πιθανοτικής τεχνικής
        p_pr = precision(prob_ranked, relevant_set)
        r_pr = recall(prob_ranked, relevant_set)
        f1_pr = f1_score(p_pr, r_pr)

        #Εκτύπωση αποτελεσμάτων στο τερματικό παράθυρο
        print("\n" + "=" * 60)
        print(f"Ερώτημα {qid} | Πλήθος Σχετικών Εγγράγων (Relevant.txt): {len(relevant_set)}")
        print(f"TF-IDF τεχνική ==> P={p_tf:.4f}, R={r_tf:.4f}, F1={f1_tf:.4f}")
        print(f"Πιθανοτική τεχνική ==> P={p_pr:.4f}, R={r_pr:.4f}, F1={f1_pr:.4f}")

        for k in K_VALUES:
            print(
                f"P@{k}: TF-IDF={precision_at_k(tfidf_ranked, relevant_set, k):.4f} "
                f"Prob={precision_at_k(prob_ranked, relevant_set, k):.4f}"
            )

        #Υπολογισμός Ακρίβειας Ανάκλησης για TF-IDF και Πιθανοτικής τεχνικής
        r_tf, p_tf = precision_recall_hits(tfidf_ranked, relevant_set)
        r_pr, p_pr = precision_recall_hits(prob_ranked, relevant_set)

        #Δημιουργία του διαγράμματος για το εκάστοτε ερώτημα με βάση
        #το τι υπολογίστηκε από την παραπάνω συνάρτηση
        outpath = plots_precision_recall_hits(r_tf, p_tf, r_pr, p_pr, qid)
        #print(f"Saved PR plot → {outpath}")

    print("Τέλος!")


if __name__ == "__main__":
    main()