# # chromaDB.py
# import os
# import tensorflow as tf
# import chromadb
# import numpy as np
# from tensorflow.keras.models import Model, load_model
# from sprite_ingestor import ingest_dataset_flat

# # Load trained model
# with open("classes.txt") as f:
#     class_names = [line.strip() for line in f]

# model = load_model("image_embedding_model.keras")
# # model.load_weights("sprite_cnn.weights.h5")

# # Embedding extractor
# embedding_model = Model(
#     inputs=model.input,
#     outputs=model.get_layer("embedding").output
# )


# # Chroma setup

# client = chromadb.PersistentClient(path="./chroma_data")

# # Check it works
# print(f"Chroma version: {chromadb.__version__}")
# print(f"Heartbeat: {client.heartbeat()}")  # Returns timestamp

# # List existing collections (should be empty initially)
# collections = client.list_collections()
# print(f"Collections: {collections}")

# collection = client.get_or_create_collection(
#     name="sprite_embeddings",
#     metadata={"distance": "cosine"}
# )

# print(f"Collection count: {collection.count()}")


# # Helper: embed one image

# def embed_image(image_path):
#     img = tf.io.read_file(image_path)
#     img = tf.image.decode_png(img, channels=3)
#     img = tf.image.resize(img, (32, 32))
#     img = tf.expand_dims(img, axis=0)

#     vec = embedding_model.predict(img, verbose=0)[0]
#     vec = vec / np.linalg.norm(vec)  # cosine-safe
#     return vec


# # EXAMPLE: add one example sprite

# sprite_path = "./Labeled_Dataset/dungeonAltars_altar_ashenzari.png" 
# sprite_id = "dungeonAltars_altar_ashenzari"

# embedding = embed_image(sprite_path)

# existing_ids = set(collection.get()["ids"])

# if sprite_id not in existing_ids:
#     collection.add(
#         ids=[sprite_id],
#         embeddings=[embedding.tolist()],
#         metadatas=[{
#             "category": "dungeonAltars",
#             "name": "altar_ashenzari"
#         }]
#     )
#     print("Sprite added.")
# else:
#     print("Sprite already exists — skipped.")



# # QUERY

# query_vector = embed_image(sprite_path)

# results = collection.query(
#     query_embeddings=[query_vector.tolist()],
#     n_results=5
# )

# print("Nearest neighbors:")
# print("IDs:", results["ids"])
# print("Distances:", results["distances"])
# print("Metadata:", results["metadatas"])
