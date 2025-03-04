import os
import pandas as pd
from tqdm import tqdm
from chunking import chunk_documents
from openai import OpenAI
from sentence_transformers import SentenceTransformer

min_token_length = 30
#embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", trust_remote_code=True) 


def createCSV(): 
    model_name_or_path=os.environ.get('EMBEDDING_MODEL', 'all-mpnet-base-v2')
    embedding_model = SentenceTransformer(model_name_or_path)
    pages_and_chunks = chunk_documents()


    openai_api_client = OpenAI()
    df = pd.DataFrame(pages_and_chunks)

    pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")


    for item in tqdm(pages_and_chunks_over_min_token_len):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"], normalize_embeddings=True)

# Turn text chunks into a single list
    text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]
# Embed all texts in batches
    text_chunk_embeddings = embedding_model.encode(text_chunks,
                                               batch_size=32, # you can use different batch sizes here for speed/performance, I found 32 works well for this use case
                                               convert_to_tensor=True) # optional to return embeddings as tensor instead of array

    text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
    embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
    text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

