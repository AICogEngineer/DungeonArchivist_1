from tensorflow.keras.models import load_model
from archivist import Archivist
from embedding_utils import images_from_paths
from data import load_unlabeled_dataset

model = load_model("image_embedding_model.keras")
archivist = Archivist()

paths, _ = load_unlabeled_dataset("./chaos_data")
X = images_from_paths(paths)

# class_names must match training
with open("classes.txt") as f:
    class_names = [l.strip() for l in f]

archivist.sort_chaos_dataset(
    model=model,
    X=X,
    image_paths=paths,
    class_names=class_names,
    output_dir="./restored_archive",
    k=5
)
