import os
import pickle
import numpy as np
from scipy.spatial.distance import cosine

FACE_DB = "data/faces"

def save_embeddings(name, embeddings):
    os.makedirs(FACE_DB, exist_ok=True)
    with open(f"{FACE_DB}/{name}.pkl", "wb") as f:
        pickle.dump(embeddings, f)

def load_embeddings():
    db = {}
    if not os.path.exists(FACE_DB):
        return db

    for file in os.listdir(FACE_DB):
        if file.endswith(".pkl"):
            with open(os.path.join(FACE_DB, file), "rb") as f:
                db[file.replace(".pkl", "")] = pickle.load(f)
    return db

def get_embedding(landmarks):
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    return points.flatten()

def match_face(embedding, database, threshold=0.35):
    best_name = None
    min_dist = 1.0

    for name, embs in database.items():
        for db_emb in embs:
            dist = cosine(embedding, db_emb)
            if dist < min_dist:
                min_dist = dist
                best_name = name

    if min_dist < threshold:
        return best_name
    return None