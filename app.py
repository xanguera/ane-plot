import os
import sys
from pathlib import Path

from absl import flags
import numpy as np
import torch
from flask import Flask, jsonify, request, render_template
from sklearn.manifold import TSNE

from acn_embed.embed.embedder.text_embedder import TextEmbedder

app = Flask(__name__)

# Define ABSL Flags
FLAGS = flags.FLAGS
if "cmudict_path" not in FLAGS:
    flags.DEFINE_string("cmudict_path", "cmudict-0.7b", "Path to the CMUdict file.")
if "host" not in FLAGS:
    flags.DEFINE_string("host", "0.0.0.0", "Host interface to bind the Flask server to.")
if "port" not in FLAGS:
    flags.DEFINE_integer("port", 5001, "Port to run the Flask server on.")
if "embeddings_cache_dir" not in FLAGS:
    flags.DEFINE_string("embeddings_cache_dir", "/tmp", "Directory to cache phonetic embeddings.")
if "default_embed_type" not in FLAGS:
    flags.DEFINE_string("default_embed_type", "phone", "Default embedding type to use (phone or grapheme).")
if "default_neighbors" not in FLAGS:
    flags.DEFINE_integer("default_neighbors", 8, "Default number of neighbors to display.")
if "ignore_sighup" not in FLAGS:
    flags.DEFINE_boolean("ignore_sighup", True, "Ignore SIGHUP signal (prevent termination when SSH connection dies).")

def _prune_by_lm_score(strings, lmscores, embeddings, lm_score_thres):
    use_idx = np.nonzero(lmscores > lm_score_thres)[0]
    strings = [strings[idx] for idx in use_idx]
    embeddings = embeddings[use_idx, :]
    return strings, lmscores, embeddings

def load_cmudict(path):
    if not path or not os.path.exists(path):
        print(f"Error: CMUdict not found at '{path}'.")
        print("Please download it by running the following command:")
        print(f"  curl -s https://raw.githubusercontent.com/Alexir/CMUdict/master/cmudict-0.7b -o {path or 'cmudict-0.7b'}")
        sys.exit(1)
        
    cmudict = {}
        
    print(f"Loading CMUdict from {path}...")
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            if line.startswith(";;;"):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            word = parts[0]
            if "(" in word and ")" in word:
                word = word.split("(")[0]
            pron = parts[1:]
            if word not in cmudict:
                cmudict[word] = pron
    print(f"Loaded {len(cmudict)} words from CMUdict.")
    return cmudict

class NNSearchBackend:
    def __init__(self, device: torch.device):
        self.device = device
        
        base_dir = Path(__file__).parent.resolve()
        embeddings_path = base_dir / "wakeword" / "embeddings-3-gram.pruned.1e-7.pt"
        strings_path = base_dir / "wakeword" / "str2score.3-gram.pruned.1e-7.pt"
        grapheme_embedder_path = base_dir / "model" / "embedder-64"
        
        # Load CMUdict
        self.cmudict = load_cmudict(FLAGS.cmudict_path)
        
        # Load strings and scores
        print("Loading strings...")
        with open(strings_path, "rb") as fobj:
            obj = torch.load(fobj, map_location="cpu", weights_only=True)
            self.strings = obj["strings"]
            self.lmscores = np.log(10) * np.array(obj["scores"])
            
        # Load embeddings (grapheme space)
        print("Loading embeddings...")
        with open(embeddings_path, "rb") as fobj:
            self.grapheme_embeddings = torch.load(fobj, map_location=device, weights_only=True).detach()
            
        # Prune
        print("Pruning by LM score...")
        self.strings, self.lmscores, self.grapheme_embeddings = _prune_by_lm_score(
            self.strings, self.lmscores, self.grapheme_embeddings, lm_score_thres=-14.0
        )
        
        print("Loading embedders...")
        self.grapheme_embedder = TextEmbedder(
            model_dir=grapheme_embedder_path, text_type="grapheme", device=device
        )
        self.phone_embedder = TextEmbedder(
            model_dir=grapheme_embedder_path, text_type="phone", device=device
        )
        
        cache_dir = Path(FLAGS.embeddings_cache_dir)
        cache_path = cache_dir / "phone_embeddings_cache.pt"
        
        loaded_from_cache = False
        if cache_path.exists():
            try:
                print(f"Loading phonetic embeddings from cache: {cache_path}...")
                self.phone_embeddings = torch.load(cache_path, map_location=device, weights_only=True)
                if self.phone_embeddings.shape[0] == len(self.strings):
                    loaded_from_cache = True
                    print("Phonetic embeddings loaded successfully from cache.")
                else:
                    print("Warning: Cached phonetic embeddings size mismatch, recomputing...")
            except Exception as e:
                print(f"Warning: Failed to load cached embeddings: {e}, recomputing...")
                
        if not loaded_from_cache:
            print("Computing phonetic embeddings for vocabulary...")
            valid_phones = list(self.phone_embedder.model.subword_to_idx.keys())
            fallback_phone = [valid_phones[0]] if valid_phones else ["AH0"]
            
            prons = []
            for word in self.strings:
                pron = self.cmudict.get(word.upper())
                if not pron:
                    pron = [c for c in word.upper()]
                # filter phonemes
                pron = [ph for ph in pron if ph in self.phone_embedder.model.subword_to_idx]
                if not pron:
                    pron = fallback_phone
                prons.append(pron)
                
            # Batch embed all vocabulary prons
            self.phone_embeddings = self.phone_embedder.get_embedding(
                prons, batch_size=500, log_interval=0
            ).detach()
            
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                print(f"Caching phonetic embeddings to {cache_path}...")
                torch.save(self.phone_embeddings, cache_path)
            except Exception as e:
                print(f"Warning: Failed to cache phonetic embeddings: {e}")
                
        print("Backend ready.")
        
    def search_and_graph(self, query_word: str, num_requested: int, embed_type: str = None):
        if embed_type is None:
            embed_type = FLAGS.default_embed_type
        query_word = query_word.upper()
        
        if embed_type == "phone":
            query_pron = self.cmudict.get(query_word)
            if not query_pron:
                query_pron = [c for c in query_word]
            query_pron = [ph for ph in query_pron if ph in self.phone_embedder.model.subword_to_idx]
            if not query_pron:
                valid_phones = list(self.phone_embedder.model.subword_to_idx.keys())
                query_pron = [valid_phones[0]] if valid_phones else ["AH0"]
                
            query_emb_tensor = self.phone_embedder.get_embedding(text=[query_pron]).detach().to(device=self.device)
            embeddings_db = self.phone_embeddings
        else:
            query_emb_tensor = self.grapheme_embedder.get_embedding(text=[query_word]).detach().to(device=self.device)
            embeddings_db = self.grapheme_embeddings
            
        # Calculate distances to all wakewords
        l2_dist = torch.sqrt(torch.sum(torch.pow(embeddings_db - query_emb_tensor, 2.0), dim=1))
        
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
                    "embedding": embeddings_db[indices[rank]].cpu().numpy()
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
    return render_template(
        "index.html",
        default_embed_type=FLAGS.default_embed_type,
        default_neighbors=FLAGS.default_neighbors
    )

@app.route("/api/search")
def api_search():
    word = request.args.get("word", "").strip()
    n = request.args.get("n", FLAGS.default_neighbors, type=int)
    embed_type = request.args.get("embed_type", FLAGS.default_embed_type).strip()
    
    if not word:
        return jsonify({"error": "word parameter is required"}), 400
        
    try:
        b = get_backend()
        graph_data = b.search_and_graph(word, n, embed_type)
        return jsonify(graph_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Parse command line flags
    FLAGS(sys.argv, known_only=True)
    
    host = FLAGS.host
    port = FLAGS.port
    
    if FLAGS.ignore_sighup:
        import signal
        if hasattr(signal, "SIGHUP"):
            print("Configured to ignore SIGHUP (hangup signal).")
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
    
    # Enable debug mode on the app object to detect it during startup
    app.debug = True
    
    # Detect SSL certificate and key
    cert_path = os.environ.get("SSL_CERT_PATH")
    key_path = os.environ.get("SSL_KEY_PATH")
    if not cert_path or not key_path:
        for c_file, k_file in [("cert.pem", "key.pem"), ("server.crt", "server.key")]:
            if os.path.exists(c_file) and os.path.exists(k_file):
                cert_path, key_path = c_file, k_file
                break

    if cert_path and key_path and os.path.exists(cert_path) and os.path.exists(key_path):
        ssl_context = (cert_path, key_path)
    else:
        ssl_context = "adhoc"
        
    # Eagerly initialize backend only in the actual server process (prevent double-loading in reloader)
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print("Initializing search backend (loading models & embeddings)...")
        get_backend()
        if isinstance(ssl_context, tuple):
            print(f" * Using custom SSL Certificate: {cert_path}")
        else:
            print(" * Using ad-hoc self-signed SSL Certificate")
        print(f" * Server is ready to receive traffic at: https://{host}:{port}")
        
    app.run(host=host, port=port, ssl_context=ssl_context)
