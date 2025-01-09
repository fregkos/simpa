"""
Authors: Bouzianas Nikoloaos, Fregkos Periklis, Gogos Lazaros
Year: 2024-2025
Task: NLP Project - Scientific Paper Search Engine based on keywords
Data & Web Science - Aristotle University of Thessaloniki
"""
# --------------------------------------- #
#  run with python main.py --limit 1000
# --------------------------------------- #
import os,sys
from pprint import pprint
import numpy as np
from config import parse_arguments
from data_loader import extract_fields, load_dataset, save_dataset
from preprocessing import create_necessary_folders, preprocess_data
from training import train_or_load_model
from metrics import find_similarity_documents

#Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted
#.\myvenv\Scripts\activate
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Restricted
 #python main.py --limit 100

def main(args):
    limit = args.limit
    fields = args.fields
    print([f for f in fields])
    # import sys
    # sys.exit()
    input_file_path = args.input_file
    dataset_file_path = args.output_file
    model_file_path = args.model_file
    clean_abstract = args.clean_abstract

    new_dataset = args.new_dataset
    new_doc2vec_model = args.new_model
    dataset = {}
    wmd_list = []

    # Firstly, create necessary directories if needed
    create_necessary_folders()

    # Check if the pruned file exists before loading it
    # And make sure the user has not asked for rebuilding the dataset from scratch
    if os.path.exists(dataset_file_path) and not new_dataset:
        dataset = load_dataset(dataset_file_path)
    else:
        # otherwise, load the full dataset and prune it
        dataset = extract_fields(input_file_path, fields, limit, clean_abstract)
        save_dataset(dataset_file_path, dataset)

    # 1. Import Doc2Vec and create a new model if it doesn't exist
    model,wmd_list = train_or_load_model(model_file_path, dataset, fields, new_doc2vec_model)

    # 2. Find top N similar papers by asking a query from the user
    top_n_results = 5
    find_similar_papers(dataset, model, top_n_results,wmd_list = wmd_list)
 
def find_similar_papers(dataset, model, top_n_results, wmd_list):
    while True:
        query = input("Enter a query: ")
        if query in ["/exit", "/e", "/quit", "/q"]:
            break
        preprocessed_query = preprocess_data(query)
        query_vector = model.infer_vector(preprocessed_query)

        vector_list = [(doc_id,model.infer_vector(tokenized_abstract)) for doc_id,tokenized_abstract in wmd_list]

        result_list = find_similarity_documents(model=model, metric='soft_cosine',wmd_list = wmd_list,vec_list=vector_list,
                                                pros_querry = preprocessed_query,topn=top_n_results,vec_quer= query_vector)
        print(len(result_list))

        for paper_id, similarity in result_list:
            print(f"Paper ID: {paper_id}, Similarity: {similarity}")
            pprint(dataset[paper_id]["title"])
            # pprint(dataset[paper_id]["abstract"])
            print("\n")
        #----------------------------------------------#
        '''
        Mε βάση τα docs της βιβλιοθήκη gensim θα έχουμε ότι:
        1) παίρνεις λέξεις απο το querry που του δίνω και φτιάχνει ένα συνολικό μέσο διάνυσμα
        2) έπειτα παίρνει ομοιότητα συνιμητόνου για αυτό το μέσο σταθμισμένο διάνυσμα --> το βάρος ΠΑΝΤΑ ΕΙΝΑΙ 1 ΓΙΑΤΙ ΤΟΥ ΔΙΝΩ ΕΝΑ DOCUMENT
        3) το μέσο διάνυσμα συγκρίνεται με
        4) το μοντέλο εκπαιδεύεται με μία λίστα απο tagged documents που μετρατρέπονται σε διανύσματα μέσω τοθ vector size που ορίζουμε εμείς επηρεάζεται κυρίως απο το vector size
        5) για κάθε διάνυασμα βρίσκεται το dot product και το cosine similarity
        5) αρχικά έχουμε ότι
        wmdistance(self, document1, document2, norm=True)  --> εχει και την word mover 
        '''
        # old code ------------------------ erase it ----------------------------#
        # print(model.dv.most_similar)
        # similar_docs = model.dv.most_similar([query_vector], topn=top_n_results)  
        # #----------------------------------------------#
        # for paper_id, similarity in similar_docs:
        #     print(f"Paper ID: {paper_id}, Similarity: {similarity}")
        #     pprint(dataset[paper_id]["title"])
        #     pprint(dataset[paper_id]["abstract"])
        #     print("\n")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
