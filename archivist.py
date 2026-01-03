import chromadb
from chromadb.config import Settings
import numpy as np
import tensorflow as tf
import os
from collections import Counter
import shutil


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
    

    def predict_knn(
    self,
    model,
    X,
    class_names,
    k=5,
    batch_size=64,
    min_confidence=0.6
    ):

        embed_model = self._get_embedding_model(model)

        embeddings = embed_model.predict(
            X,
            batch_size=batch_size,
            verbose=1
        )

        predictions = []
        confidences = []

        for emb in embeddings:
            results = self.collection.query(
                query_embeddings=[emb.tolist()],
                n_results=k,
                where={"split": "train"}
            )

            neighbor_labels = [
                m["class_name"] for m in results["metadatas"][0]
            ]

            counts = Counter(neighbor_labels)
            pred_class, count = counts.most_common(1)[0]
            confidence = count / k

            if confidence >= min_confidence:
                predictions.append(pred_class)
            else:
                predictions.append("uncertain")

            confidences.append(confidence)

        return predictions, confidences
    
    def sort_chaos_dataset(
        self,
        model,
        X,
        image_paths,
        class_names,
        output_dir,
        k=5,
        min_knn_confidence=0.6,
        min_softmax_confidence=0.6,
        batch_size=64
    ):
        
        #Sort unlabeled images using hybrid Softmax + kNN predictions.

    
        # Accept label if kNN == softmax
        # OR if kNN confidence >= min_knn_confidence
        # Else label as 'uncertain'

        os.makedirs(output_dir, exist_ok=True)

        # --- Models ---
        embed_model = self._get_embedding_model(model)

        # --- Predictions ---
        embeddings = embed_model.predict(
            X,
            batch_size=batch_size,
            verbose=1
        )

        softmax_preds = model.predict(
            X,
            batch_size=batch_size,
            verbose=1
        )

        results_summary = []

        for i, emb in enumerate(embeddings):
            # ---- kNN ----
            knn_results = self.collection.query(
                query_embeddings=[emb.tolist()],
                n_results=k,
                where={"split": "train"}
            )

            neighbor_labels = [
                m["class_name"] for m in knn_results["metadatas"][0]
            ]

            knn_counts = Counter(neighbor_labels)
            knn_label, knn_votes = knn_counts.most_common(1)[0]
            knn_conf = knn_votes / k

            # ---- Softmax ----
            sm_probs = softmax_preds[i]
            sm_idx = int(np.argmax(sm_probs))
            sm_label = class_names[sm_idx]
            sm_conf = float(sm_probs[sm_idx])

            # ---- Hybrid Decision ----
            if knn_label == sm_label:
                final_label = knn_label
                reason = "agreement"
            elif knn_conf >= min_knn_confidence:
                final_label = knn_label
                reason = "knn_confident"
            elif sm_conf >= min_softmax_confidence:
                final_label = sm_label
                reason = "softmax_confident"
            else:
                final_label = "uncertain"
                reason = "low_confidence"

            # ---- Copy Image ----
            target_dir = os.path.join(output_dir, final_label)
            os.makedirs(target_dir, exist_ok=True)

            shutil.copy(
                image_paths[i],
                os.path.join(target_dir, os.path.basename(image_paths[i]))
            )

            results_summary.append({
                "image": os.path.basename(image_paths[i]),
                "label": final_label,
                "knn_label": knn_label,
                "knn_conf": knn_conf,
                "softmax_label": sm_label,
                "softmax_conf": sm_conf,
                "decision": reason
            })

        print(f"[Archivist] Chaos dataset sorted into '{output_dir}'")
        return results_summary