import os
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, render_template
from sklearn.manifold import TSNE

from acn_embed.embed.embedder.text_embedder import TextEmbedder

app = Flask(__name__)

def _prune_by_lm_score(strings, lmscores, embeddings, lm_score_thres):
    use_idx = np.nonzero(lmscores > lm_score_thres)[0]
    strings = [strings[idx] for idx in use_idx]
    embeddings = embeddings[use_idx, :]
    return strings, lmscores, embeddings

class NNSearchBackend:
    def __init__(self, device: torch.device):
        self.device = device
        
        base_dir = Path(__file__).parent.resolve()
        embeddings_path = base_dir / "wakeword" / "embeddings-3-gram.pruned.1e-7.pt"
        strings_path = base_dir / "wakeword" / "str2score.3-gram.pruned.1e-7.pt"
        grapheme_embedder_path = base_dir / "model" / "embedder-64"
        
        # Load strings and scores
        print("Loading strings...")
        with open(strings_path, "rb") as fobj:
            obj = torch.load(fobj, map_location="cpu", weights_only=True)
            self.strings = obj["strings"]
            self.lmscores = np.log(10) * np.array(obj["scores"])
            
        # Load embeddings
        print("Loading embeddings...")
        with open(embeddings_path, "rb") as fobj:
            self.embeddings = torch.load(fobj, map_location=device, weights_only=True).detach()
            
        # Prune
        print("Pruning by LM score...")
        self.strings, self.lmscores, self.embeddings = _prune_by_lm_score(
            self.strings, self.lmscores, self.embeddings, lm_score_thres=-14.0
        )
        
        print("Loading embedder...")
        self.embedder = TextEmbedder(
            model_dir=grapheme_embedder_path, text_type="grapheme", device=device
        )
        print("Backend ready.")
        
    def search_and_graph(self, query_word: str, num_requested: int):
        query_word = query_word.upper()
        
        # Get query embedding
        query_emb_tensor = self.embedder.get_embedding(text=[query_word]).detach().to(device=self.device)
        
        # Calculate distances to all wakewords
        l2_dist = torch.sqrt(torch.sum(torch.pow(self.embeddings - query_emb_tensor, 2.0), dim=1))
        
        # Retrieve top k (ask for more to filter out exact query matches)
        values, indices = torch.topk(l2_dist, dim=0, k=num_requested * 2, largest=False, sorted=True)
        
        neighbors = []
        shown = 0
        rank = 0
        while shown < num_requested and rank < len(values):
            dist = values[rank].item()
            result_string = self.strings[indices[rank]]
            if result_string != query_word:
                neighbors.append({
                    "word": result_string,
                    "distance": dist,
                    "embedding": self.embeddings[indices[rank]].cpu().numpy()
                })
                shown += 1
            rank += 1
            
        # Prepare graph nodes: Query + Neighbors
        nodes = [{"id": query_word, "is_query": True}]
        for n in neighbors:
            nodes.append({"id": n["word"], "is_query": False, "dist_to_query": n["distance"]})
            
        # Prepare embeddings for t-SNE
        all_embeddings = [query_emb_tensor.cpu().numpy()[0]] + [n["embedding"] for n in neighbors]
        X = np.array(all_embeddings)
        
        # Run t-SNE to get 2D positions
        n_samples = X.shape[0]
        perplexity = min(5.0, n_samples - 1.0) if n_samples > 1 else 1.0
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
        if n_samples > 1:
            X_2d = tsne.fit_transform(X)
        else:
            X_2d = np.zeros((1, 2))
            
        for i, node in enumerate(nodes):
            node["x"] = float(X_2d[i, 0])
            node["y"] = float(X_2d[i, 1])
            
        # Compute edges (fully connected graph based on pairwise distances)
        edges = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                weight = 1.0 / (1.0 + float(dist)) # convert distance to similarity weight
                edges.append({
                    "source": nodes[i]["id"],
                    "target": nodes[j]["id"],
                    "weight": weight,
                    "distance": float(dist)
                })
                
        return {"nodes": nodes, "edges": edges}

# Initialize backend globally so it loads models on server start
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backend = None

def get_backend():
    global backend
    if backend is None:
        backend = NNSearchBackend(device)
    return backend

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/search")
def api_search():
    word = request.args.get("word", "").strip()
    n = request.args.get("n", 8, type=int)
    
    if not word:
        return jsonify({"error": "word parameter is required"}), 400
        
    try:
        b = get_backend()
        graph_data = b.search_and_graph(word, n)
        return jsonify(graph_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
