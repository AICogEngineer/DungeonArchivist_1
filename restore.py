import os
import shutil
import numpy as np
import tensorflow as tf
import chromadb
from tensorflow.keras.models import load_model, Model

CHAOS_DIR = "./chaos_data"
OUTPUT_DIR = "./restored_archive"
REVIEW_DIR = "./review_pile"

K = 5
DIST_THRESHOLD = 0.35
CONF_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.15

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)

# Load model
model = load_model("image_embedding_model.keras")

embedding_model = Model(
    inputs=model.input,
    outputs=model.get_layer("embedding").output
)

# Load Chroma
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_collection("sprite_embeddings")

def embed_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (32, 32))
    img = tf.expand_dims(img, 0)
    vec = embedding_model.predict(img, verbose=0)[0]
    return vec / np.linalg.norm(vec)

def coarse_label(label):
    if "weapon" in label.lower():
        return "Weapon"
    if "wall" in label.lower():
        return "Environment"
    return "Misc"

def weighted_vote(labels, distances):
    scores = {}
    for l, d in zip(labels, distances):
        scores[l] = scores.get(l, 0) + (1 - d)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

for fname in os.listdir(CHAOS_DIR):
    if not fname.endswith(".png"):
        continue

    src = os.path.join(CHAOS_DIR, fname)

    try:
        vec = embed_image(src)

        res = collection.query(
            query_embeddings=[vec.tolist()],
            n_results=K
        )

        labels = [coarse_label(m["category"]) for m in res["metadatas"][0]]
        distances = res["distances"][0]

        ranked = weighted_vote(labels, distances)

        top_label, top_score = ranked[0]
        margin = top_score - (ranked[1][1] if len(ranked) > 1 else 0)
        mean_dist = np.mean(distances)

        confident = (
            top_score >= CONF_THRESHOLD and
            mean_dist <= DIST_THRESHOLD and
            margin >= MARGIN_THRESHOLD
        )

        if not confident:
            shutil.move(src, os.path.join(REVIEW_DIR, fname))
            continue

        dest = os.path.join(OUTPUT_DIR, top_label)
        os.makedirs(dest, exist_ok=True)
        shutil.move(src, os.path.join(dest, fname))

    except Exception:
        shutil.move(src, os.path.join(REVIEW_DIR, fname))

print("Restoration complete.")
