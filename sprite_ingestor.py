# sprite_ingestor.py
import os


def ingest_dataset_flat(
    root_dir,
    collection,
    embed_image_fn,
    batch_size=32
):
    """
    Ingest all sprites from a flat folder into an existing Chroma collection.

    Parameters:
    - root_dir: path to Labeled_Dataset
    - collection: already-created Chroma collection
    - embed_image_fn: function(image_path) -> embedding vector
    """

    image_paths = [
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.lower().endswith(".png")
    ]

    print(f"Found {len(image_paths)} sprites")

    existing_ids = set(collection.get()["ids"])

    ids, embeddings, metadatas = [], [], []

    for path in image_paths:
        filename = os.path.basename(path)

        try:
            base = os.path.splitext(filename)[0]
            category, name = base.split("_", 1)
            sprite_id = f"{category}_{name}"

            if sprite_id in existing_ids:
                continue

            embedding = embed_image_fn(path)

            ids.append(sprite_id)
            embeddings.append(embedding.tolist())
            metadatas.append({
                "category": category,
                "name": name,
                "filename": filename
            })

            if len(ids) >= batch_size:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                existing_ids.update(ids)
                ids, embeddings, metadatas = [], [], []

        except Exception as e:
            print(f"Failed on {filename}: {e}")

    # Flush remaining
    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

    print("Dataset ingestion complete.")
