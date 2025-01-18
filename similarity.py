import pprint

from gensim.models.doc2vec import Doc2Vec
from keybert import KeyBERT

from training.preprocessing import preprocess_data


def find_similar_papers(
    dataset: dict,
    model: Doc2Vec,
    top_n_results: int,
):

    while True:
        query = input("Enter a query: ")
        if query in ["/exit", "/e", "/quit", "/q"]:
            break
        preprocessed_query = preprocess_data(query)
        query_vector = model.infer_vector(preprocessed_query)

        similar_docs = model.dv.most_similar([query_vector], topn=top_n_results)
        print(f"{similar_docs=}")

        # Example
        # Compare a paper with it's categories
        print("\n\n\nExample: Compare a paper with it's categories")
        paper_id = "0704.0001"
        paper_vs_self_categoies = model.dv.similarity(
            paper_id, dataset[paper_id]["categories"]
        )
        print(
            f"Paper ID: {paper_id},\ncategories: {dataset[paper_id]['categories']},\nSimilarity to self categories: {paper_vs_self_categoies}"
        )
        print("\n" * 3)

        for paper_id, similarity in similar_docs:
            try:
                print(f"Paper ID: {paper_id}, Similarity: {similarity}")
                pprint(dataset[paper_id]["title"])
                pprint(dataset[paper_id]["abstract"])
                print("\n")

            except KeyError:
                print(
                    f"The category {paper_id} is {similarity} similar to your query.\n"
                )
                continue

        keybert_model = KeyBERT()

        query_keywords = keybert_model.extract_keywords(
            query,
            keyphrase_ngram_range=(1, 3),
        )

        print("Extracted keywords, based on your query:")
        for keyword, similarity_to_query in query_keywords:
            # delimited by spaces instead of new lines for easy copy-pasting
            print(keyword, end=" ")
        print()  # print empty new line
