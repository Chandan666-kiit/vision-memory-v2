import faiss
import numpy as np
import os
import pickle

INDEX_FILE = "face_index.faiss"
META_FILE = "face_meta.pkl"

# dimension of face_recognition embeddings
DIM = 128


class FaceVectorDB:

    def __init__(self):

        if os.path.exists(INDEX_FILE):

            self.index = faiss.read_index(INDEX_FILE)

            with open(META_FILE, "rb") as f:
                self.names = pickle.load(f)

        else:

            self.index = faiss.IndexFlatL2(DIM)

            self.names = []

    def save(self):

        faiss.write_index(self.index, INDEX_FILE)

        with open(META_FILE, "wb") as f:
            pickle.dump(self.names, f)

    def add_face(self, name, embedding):

        vec = np.array([embedding]).astype("float32")

        self.index.add(vec)

        self.names.append(name)

        self.save()

    def search(self, embedding):

        if self.index.ntotal == 0:
            return None

        vec = np.array([embedding]).astype("float32")

        D, I = self.index.search(vec, 1)

        if D[0][0] < 0.6:
            return self.names[I[0][0]]

        return None


face_db = FaceVectorDB()