#def most_similar_to_given(self, key1, keys_list):
#        """Get the `key` from `keys_list` most similar to `key1`."""
#       return keys_list[argmax([self.similarity(key1, key) for key in keys_list])]
from sklearn.metrics import jaccard_score
import sys
import numpy as np
from scipy.spatial.distance import cosine
from gensim.models.doc2vec import Doc2Vec
from gensim import corpora
from gensim.similarities import WordEmbeddingSimilarityIndex,SparseTermSimilarityMatrix

   # TODO 3 Levennsthtein library


def calc_jaccard_simillarity(string_querry,string_doc):
    'returns the jaccard simillarity of a given querry with different abstracts'
    set1= set(string_querry)
    set2 = set(string_doc)
    inter_sect_nominator = set1.intersection(set2)
    union_denominator = set1.union(set2)
    return len(inter_sect_nominator)/len(union_denominator)
    
def calc_cosine_similarity(querry_vec,vec2):
    '''
    Insert a vector list and return cosine similarity
    Idealy we want the bigger values
    '''
    return 1 - cosine(np.array(querry_vec),np.array(vec2))
def calc_euclidean_distance(array1,array2):
    '''
    Takes two arrays and return the euclidean distance
    ! Note --> this is distance so we want minimal values 
    '''
    return np.sqrt(np.sum(array1-array2)**2)
def calc_soft_cosine_similarity():
    # yOU NEED A SIMILARITY MATRIX
    #
    pass

def identify_gensim_model(in_model):
    string_to_check=str(type(in_model))
    if 'gensim' in  string_to_check:
        if '2vec' in string_to_check.lower():
            if 'doc' in string_to_check:
                return True,False
            elif 'word' in string_to_check:
                return False,True
        else:
            raise ValueError("cannot identify if that is a 2VEC model")
    else:
        raise ValueError("cannot handle a non gensim model")


def find_similarity_documents(**kwargs):
    '''
    This function returns similarity metric between two documents
    Input: User can insert the implemented model for embedding acquisition
    Output: A float number which is the similarity between the two documents
    metric,model, documents_strings, user_querry, abstract_docs , list of strings that contain abstract ==tokenized_list
    '''
    import sys
    print("inside BOUZISSS CODE")
    jaccard_list,euclidean_list,soft_cosine_list,cosine_list,wmd_distance_list,sorted_results =[],[],[],[],[],[]
    model_is_doc_2_vec ,model_is_word_2_vec = False,False # Only for gensim models
    model_is_doc_2_vec,model_is_word_2_vec = identify_gensim_model(in_model=kwargs["model"])
    # todo you need model training corpus to have a list of trained models
    if "soft" in kwargs["metric"].lower() and 'cosine' in  kwargs["metric"].lower():
        #https://radimrehurek.com/gensim/auto_examples/tutorials/run_scm.html
        # https://www.geeksforgeeks.org/nlp-gensim-tutorial-complete-guide-for-beginners/
        # https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb
        if model_is_doc_2_vec:
            soft_cosine_docs_list =[w[1] for w in kwargs['wmd_list']]
            dicton_corpus = corpora.Dictionary(soft_cosine_docs_list) # πρέπει να βάλω εδώ όλα τα documents
            final_soft_cosine_list = [(doc_id,dicton_corpus.doc2bow(docu)) for doc_id,docu in kwargs['wmd_list']]
            similarity_index = WordEmbeddingSimilarityIndex(kwargs['model'].wv)
            similarity_matrix = SparseTermSimilarityMatrix(similarity_index,dicton_corpus)
            f_querry =dicton_corpus.doc2bow(kwargs["pros_querry"])
      
            soft_similarity = [(d_id,float(similarity_matrix.inner_product(f_querry,doc, normalized=(True,True)))) for d_id,doc in final_soft_cosine_list]
            if not any(isinstance(val[1],float) for val in soft_similarity):            
                        raise ValueError('error with cosine similairty')
            
            sorted_results =  sorted(soft_similarity,key=lambda x:x[1],reverse= True)
            return  sorted_results[0:kwargs["topn"]]

    elif 'cosine' in  kwargs["metric"].lower():
        if model_is_doc_2_vec:
            cosine_list= [(ve[0],calc_cosine_similarity(querry_vec=kwargs['vec_quer'],vec2=ve[1])) for ve in kwargs['vec_list']] # τις παρόμοιες λέξεις τις κάνει hanlde το μοντέλο
            if not any(isinstance(val[1],float) for val in cosine_list):
                    raise ValueError('error with cosine similairty')
            sorted_results =  sorted(cosine_list,key=lambda x:x[1],reverse= True)
            return  sorted_results[0:kwargs["topn"]]
    elif 'euclidean' in kwargs['metric'].lower():
        if model_is_doc_2_vec:
            euclidean_list= [(ve[0],calc_euclidean_distance(array1=kwargs['vec_quer'],array2=ve[1])) for ve in kwargs['vec_list']] # τις παρόμοιες λέξεις τις κάνει hanlde το μοντέλο
            if not any(isinstance(val[1],float) for val in euclidean_list):
                    raise ValueError('error with euclidean distance')
            sorted_results =  sorted(euclidean_list,key=lambda x:x[1])
            return  sorted_results[0:kwargs["topn"]]
    elif 'jaccard' in kwargs['metric'].lower():
        if model_is_doc_2_vec:
            jaccard_list = [(doc[0],calc_jaccard_simillarity(string_querry=kwargs["pros_querry"],string_doc=doc[1])) for doc in kwargs['wmd_list']] # τις παρόμοιες λέξεις τις κάνει hanlde το μοντέλο
            if not any(isinstance(val[1],float) for val in jaccard_list):
                    raise ValueError('error with jaccard similarity')
            sorted_results =  sorted(jaccard_list,key=lambda x:x[1],reverse= True)
            return  sorted_results[0:kwargs["topn"]]
        
    elif 'wmd' in kwargs['metric'].lower():
        # requires list of all tokenized abstracts and the tokenized abstract
        if model_is_doc_2_vec:
            wmd_distance_list=[(abstract_doc[0],kwargs['model'].wv.wmdistance(kwargs["pros_querry"], abstract_doc[1])) for doc_index,abstract_doc in enumerate(kwargs["wmd_list"])] 
            if not any(isinstance(val[1],float) for val in wmd_distance_list):
                raise ValueError('error with word mover distance')
            sorted_results = sorted(wmd_distance_list,key=lambda x:x[1])
            return  sorted_results[0:kwargs["topn"]] # the first value is the same paper because we are not excluding it from the dataset
        elif model_is_word_2_vec:
            raise ValueError("this is unaccceptable model")
        
        


    # ------------ ένα θέμα που προκύπτει είναι πώς θα βρω μία διανυσματική αναπαράσταση με βάση τα embeddings 
    # δηλαδή θα πρέπει να βρω ένα τρόπο να αναπαραστήσωτην παράγραφο σε ένα συνολικό διάνυσμα για να πάρω το cosine



    #Jaccard
    #Mikowski --> oxi 
    # Euclidean
    #
    #

def evaluate_model():
    # different similarrity measures
    # See paper nature 
    # 1) DCG
    # 2) MAP
    pass