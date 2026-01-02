import chromadb
from chromadb.config import Settings
import numpy as np
import tensorflow as tf
import os
from collections import Counter


class Archivist:
    def __init__(
        self,
        persist_dir="chroma_db",
        collection_name="image_embeddings"
    ):
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _get_embedding_model(self, model):
        embedding_output = model.get_layer("embedding").output
        return tf.keras.Model(
            inputs=model.input,
            outputs=embedding_output
        )

    def _make_ids(self, split, class_names, y):
        """
        Deterministic IDs so embeddings are updated instead of duplicated.
        """
        return [
            f"{split}_{class_names[int(label)]}_{i}"
            for i, label in enumerate(y)
        ]

    def store_embeddings(
        self,
        model,
        X,
        y,
        class_names,
        split="train",
        batch_size=64
    ):
        """
        Upserts embeddings using deterministic IDs.
        Safe to call multiple times.
        """
        assert split in {"train", "val", "test", "inference"}

        embed_model = self._get_embedding_model(model)

        embeddings = embed_model.predict(
            X,
            batch_size=batch_size,
            verbose=1
        ).astype(np.float32)

        ids = self._make_ids(split, class_names, y)

        metadatas = [
            {
                "split": split,
                "class_name": class_names[int(label)]
            }
            for label in y
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        print(f"[Archivist] Upserted {len(embeddings)} embeddings ({split})")

    def evaluate_knn(
        self,
        model,
        X_val,
        y_val,
        class_names,
        k=5,
        batch_size=64
    ):
        """
        k-NN evaluation using training embeddings only.
        """
        embed_model = self._get_embedding_model(model)

        embeddings = embed_model.predict(
            X_val,
            batch_size=batch_size,
            verbose=1
        )

        correct = 0

        for i, emb in enumerate(embeddings):
            results = self.collection.query(
                query_embeddings=[emb.tolist()],
                n_results=k,
                where={"split": "train"}
            )

            neighbor_labels = [
                m["class_name"] for m in results["metadatas"][0]
            ]

            pred_class = Counter(neighbor_labels).most_common(1)[0][0]
            true_class = class_names[int(y_val[i])]

            if pred_class == true_class:
                correct += 1

        accuracy = correct / len(X_val)
        print(f"[Archivist] k-NN accuracy (k={k}): {accuracy:.4f}")
        return accuracy