import sys

from augmentation import augmentation
from generation import generation
from retrieval import get_similar_texts
from retrieval import load_chunks_with_embeddings
from dotenv import load_dotenv

load_dotenv()

if len(sys.argv) < 2:
    print("Usage: python3 rag.py <question>")
    sys.exit(1)

question = sys.argv[1]
embeddings, pages_and_chunks = load_chunks_with_embeddings()

relevant_chunks = get_similar_texts(question, embeddings, pages_and_chunks)
prompts = augmentation(question, relevant_chunks)
answer = generation(prompts)

print('***** Answer from LLM *****')
print(answer)

if 'I could not find an answer to your question.' not in answer:

    print('For more details, please refer to the following documents:')
    for chunk in relevant_chunks:
        print(chunk)