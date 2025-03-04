
import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import util, SentenceTransformer
from openai import OpenAI
from embeddings import createCSV

createCSV()
openai_api_client = OpenAI()


def load_chunks_with_embeddings() -> list[dict]:


    # Import texts and embedding df
    text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_df.csv")

    # Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

    # Convert texts and embedding df to list of dicts
    pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

    # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
    embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32)
    
    print("Shape of embeddings:", embeddings.shape)
    return embeddings, pages_and_chunks



'''

def perform_vector_similarity(user_question_embedding: list[float],
                              stored_chunk_embeddings: list[dict]) -> list[dict]:
    chunks_with_similarity_score = [
        (
            numpy.dot(numpy.array(chunk['embeddings']), numpy.array(user_question_embedding)),
            chunk
        )
        for chunk in stored_chunk_embeddings
    ]

    chunks_sorted_by_similarity = sorted(chunks_with_similarity_score, reverse=True, key=lambda score: score[0])

    return [chunk_with_similarity[1] for chunk_with_similarity in chunks_sorted_by_similarity]
'''


def dot_product(vector1, vector2):
    return torch.dot(vector1, vector2)

def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)

    # Get Euclidean/L2 norm of each vector (removes the magnitude, keeps direction)
    norm_vector1 = torch.sqrt(torch.sum(vector1**2))
    norm_vector2 = torch.sqrt(torch.sum(vector2**2))

    return dot_product / (norm_vector1 * norm_vector2)

def retrieval(query: str, n_resources_to_return: int = 5 ):
    model_name=os.environ.get('EMBEDDING_MODEL', 'all-mpnet-base-v2')
    #response = openai_api_client.embeddings.create(input=user_question, model=model)
    #embedding_from_user_question = response.data[0].embedding
    sentence_model = SentenceTransformer(model_name)
    query_embedding = sentence_model.encode(query) 
    embeddings, pages_and_chunks= load_chunks_with_embeddings()
  
    #all_chunks_with_similarity_score = perform_vector_similarity(embedding_from_user_question,stored_chunk_embeddings)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    #three_most_relevant_chunk = all_chunks_with_similarity_score[:TOP_NUMBER_OF_CHUNKS_TO_RETRIEVE]
    scores, indices = torch.topk(input=dot_scores, 
                                 k=n_resources_to_return)
    print(scores)
    return scores, indices

def get_similar_texts(query: str, embeddings: torch.tensor,
                                 pages_and_chunks: list[dict],
                                 n_resources_to_return: int=5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.

    Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
    """
    
    scores, indices = retrieval(query=query, n_resources_to_return=n_resources_to_return)
    
    # print(f"Query: {query}\n")
    # print("Results:")
    similar_text = []
    # Loop through zipped together scores and indicies
    for score, index in zip(scores, indices):
        # print(f"Score: {score:.4f}")
        # # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        # print(pages_and_chunks[index]["sentence_chunk"])
        similar_text.append(pages_and_chunks[index]["sentence_chunk"])
        # Print the page number too so we can reference the textbook further and check the results
        #print(f"Page number: {pages_and_chunks[index]['page_number']}")
        #print("\n")
    return similar_text

retrieval("warfighting function")