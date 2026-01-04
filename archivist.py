import os
import shutil
import numpy as np
import tensorflow as tf
import chromadb
from collections import Counter
from embedding_utils import load_and_preprocess_image

class Archivist:
    def __init__(self, persist_dir="chroma_db", collection_name="image_embeddings"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _get_embedding_model(self, model):
        return tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer("embedding").output
        )

    def store_embeddings(self, model, X, y, class_names, split="train", batch_size=32):
        embed_model = self._get_embedding_model(model)
        embeddings = embed_model.predict(X, batch_size=batch_size, verbose=1)

        ids = [f"{split}_{i}" for i in range(len(y))]
        metadatas = [
            {"split": split, "class_name": class_names[int(label)]}
            for label in y
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        print(f"[Archivist] Stored {len(ids)} embeddings ({split})")

    # ---------- MERGED DECISION LOGIC ----------
    def decide_label(
        self,
        knn_labels,
        knn_distances,
        softmax_probs,
        class_names,
        k,
        dist_threshold=0.30,
        margin_threshold=0.10,
        min_softmax_conf=0.6
    ):
        # Weighted kNN (your logic)
        scores = {}
        for lbl, d in zip(knn_labels, knn_distances):
            scores[lbl] = scores.get(lbl, 0) + (1 - d)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_knn, top_score = ranked[0]
        margin = top_score - (ranked[1][1] if len(ranked) > 1 else 0)
        mean_dist = float(np.mean(knn_distances))

        knn_confident = (mean_dist < dist_threshold and margin > margin_threshold)

        # Softmax (their logic)
        sm_idx = int(np.argmax(softmax_probs))
        sm_label = class_names[sm_idx]
        sm_conf = float(softmax_probs[sm_idx])

        if knn_confident and top_knn == sm_label:
            return top_knn, True, "knn+softmax_agree"
        if knn_confident:
            return top_knn, True, "knn_geometry"
        if sm_conf >= min_softmax_conf:
            return sm_label, True, "softmax_rescue"

        return "uncertain", False, "low_confidence"

    # ---------- CHAOS SORTING ----------
    def sort_chaos_dataset(
        self,
        model,
        X,
        image_paths,
        class_names,
        output_dir,
        k=5,
        batch_size=32
    ):
        os.makedirs(output_dir, exist_ok=True)

        embed_model = self._get_embedding_model(model)
        embeddings = embed_model.predict(X, batch_size=batch_size, verbose=1)
        softmax_preds = model.predict(X, batch_size=batch_size, verbose=1)

        for i, emb in enumerate(embeddings):
            res = self.collection.query(
                query_embeddings=[emb.tolist()],
                n_results=k,
                where={"split": "train"}
            )

            knn_labels = [m["class_name"] for m in res["metadatas"][0]]
            knn_distances = res["distances"][0]

            final_label, confident, reason = self.decide_label(
                knn_labels,
                knn_distances,
                softmax_preds[i],
                class_names,
                k
            )

            dest = "review_pile" if not confident else final_label
            target_dir = os.path.join(output_dir, dest)
            os.makedirs(target_dir, exist_ok=True)

            shutil.copy(
                image_paths[i],
                os.path.join(target_dir, os.path.basename(image_paths[i]))
            )

            print(f"{image_paths[i]} → {final_label} ({reason})")
