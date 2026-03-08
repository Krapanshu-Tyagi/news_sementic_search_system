import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataloader import load_data
from embeddings.embedder import Embedder

def main():
    docs, labels = load_data()
    print("Total documents:", len(docs))
    print("Total labels:", len(set(labels)))

    embedder = Embedder()

    print("\nGenerating embeddings...\n")

    embeddings = embedder.embed(docs[:1000])   # test with 1000 first

    print("\nEmbedding shape:", embeddings.shape)


if __name__ == "__main__":
    main()