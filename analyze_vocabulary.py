import os
import sys
# pyrefly: ignore [missing-import]
import torch
import numpy as np
from pathlib import Path
from absl import app, flags

# Add submodule path to resolve acn_embed
sys.path.append(os.path.join(os.path.dirname(__file__), "ml-acn-embed", "src"))
from acn_embed.embed.embedder.text_embedder import TextEmbedder

FLAGS = flags.FLAGS
if "num_neighbors" not in FLAGS:
    flags.DEFINE_integer("num_neighbors", 8, "Number of nearest neighbors to consider.")
if "cmudict_path" not in FLAGS:
    flags.DEFINE_string("cmudict_path", "cmudict-0.7b", "Path to the CMUdict file.")
if "vocab_path" not in FLAGS:
    flags.DEFINE_string("vocab_path", "wakeword/str2score.3-gram.pruned.1e-7.pt", "Path to pruned vocabulary file.")
if "embeddings_cache_dir" not in FLAGS:
    flags.DEFINE_string("embeddings_cache_dir", "/tmp", "Directory to cache phonetic embeddings.")
if "use_full_cmudict" not in FLAGS:
    flags.DEFINE_boolean("use_full_cmudict", False, "Use the entire CMUdict instead of the pruned vocabulary.")

def load_cmudict(path):
    if not path or not os.path.exists(path):
        print(f"Error: CMUdict not found at '{path}'.")
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
            prons = parts[1:]
            cmudict[word.upper()] = prons
    return cmudict

def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load CMUdict
    cmudict = load_cmudict(FLAGS.cmudict_path)
    
    # 2. Determine vocabulary list
    if FLAGS.use_full_cmudict:
        strings = sorted(list(cmudict.keys()))
        print(f"Using full CMUdict vocabulary: {len(strings):,d} words.")
    else:
        if not os.path.exists(FLAGS.vocab_path):
            print(f"Error: Vocabulary file '{FLAGS.vocab_path}' not found.")
            sys.exit(1)
        print(f"Loading pruned vocabulary from {FLAGS.vocab_path}...")
        strings_dict = torch.load(FLAGS.vocab_path, weights_only=False)
        strings = sorted([w.upper() for w in strings_dict.keys()])
        print(f"Using pruned vocabulary: {len(strings):,d} words.")
        
    # 3. Load or compute phonetic embeddings
    cache_dir = Path(FLAGS.embeddings_cache_dir)
    cache_name = "full_cmudict_phone_embeddings.pt" if FLAGS.use_full_cmudict else "pruned_phone_embeddings.pt"
    cache_path = cache_dir / cache_name
    
    # Resolve the model directory (expects a Path object pointing to the directory containing model files)
    base_dir = Path(__file__).parent
    model_dir = base_dir / "model" / "embedder-64"
    if not model_dir.exists():
        model_dir = base_dir / "ml-acn-embed" / "model" / "embedder-64"
        
    phone_embedder = TextEmbedder(
        model_dir=model_dir, text_type="phone", device=device
    )
    
    embeddings = None
    if cache_path.exists():
        try:
            print(f"Loading embeddings from cache: {cache_path}...")
            embeddings = torch.load(cache_path, map_location=device, weights_only=True)
            if embeddings.shape[0] != len(strings):
                print("Cache shape mismatch, recomputing...")
                embeddings = None
        except Exception as e:
            print(f"Warning: failed to load cache: {e}")
            embeddings = None
            
    if embeddings is None:
        print("Computing phonetic embeddings...")
        valid_phones = list(phone_embedder.model.subword_to_idx.keys())
        fallback_phone = [valid_phones[0]] if valid_phones else ["AH0"]
        
        prons = []
        for word in strings:
            pron = cmudict.get(word)
            if not pron:
                pron = [c for c in word]
            pron = [ph for ph in pron if ph in phone_embedder.model.subword_to_idx]
            if not pron:
                pron = fallback_phone
            prons.append(pron)
            
        embeddings = phone_embedder.get_embedding(
            prons, batch_size=1000, log_interval=0
        ).detach().to(device)
        
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Caching embeddings to {cache_path}...")
            torch.save(embeddings, cache_path)
        except Exception as e:
            print(f"Warning: failed to cache: {e}")
            
    # Move embeddings to target device
    embeddings = embeddings.to(device)
    n_words = len(strings)
    if n_words <= 1:
        print("Error: Vocabulary must have at least 2 words to compute neighbors.")
        sys.exit(1)
        
    actual_k = min(FLAGS.num_neighbors, n_words - 1)
    k = actual_k + 1  # +1 because the word itself is its own closest neighbor
    
    print(f"Computing nearest neighbors (considering {actual_k} neighbors per word) in batches...")
    avg_dists = np.zeros(n_words)
    nearest_1st_dists = np.zeros(n_words)
    
    # Process in batches to avoid memory allocation limit
    batch_size = 500
    for i in range(0, n_words, batch_size):
        end_idx = min(i + batch_size, n_words)
        batch_embs = embeddings[i:end_idx]
        
        # Pairwise distance from batch to all embeddings
        dists = torch.cdist(batch_embs, embeddings, p=2.0)
        
        # Sort distances (k smallest values)
        values, _ = torch.topk(dists, k=k, dim=1, largest=False, sorted=True)
        
        # The first column is distance=0.0 (the word itself), so we take columns 1 to k
        neighbor_dists = values[:, 1:k].cpu().numpy()
        
        avg_dists[i:end_idx] = np.mean(neighbor_dists, axis=1)
        nearest_1st_dists[i:end_idx] = neighbor_dists[:, 0]
        
        if (i // batch_size) % 10 == 0:
            print(f"Processed {end_idx:,d} / {n_words:,d} words...")
            
    # Sort results
    sorted_by_density = np.argsort(avg_dists)  # smallest avg distance first
    sorted_by_isolation = np.argsort(nearest_1st_dists)[::-1]  # largest nearest 1st distance first
    sorted_by_avg_isolation = np.argsort(avg_dists)[::-1]  # largest average distance first
    
    num_to_display = min(20, n_words)
    
    print("\n" + "="*80)
    print(f"TOP {num_to_display} WORDS WITH THE SMALLEST AVERAGE DISTANCE TO NEAREST {actual_k} WORDS (DENSE REGIONS)")
    print("="*80)
    for idx in range(num_to_display):
        w_idx = sorted_by_density[idx]
        print(f"{idx+1:2d}. {strings[w_idx]:<18} (Avg distance to nearest {actual_k}: {avg_dists[w_idx]:.4f})")
        
    print("\n" + "="*80)
    print(f"TOP {num_to_display} WORDS WITH THE LARGEST DISTANCE TO THEIR NEAREST NEIGHBOR (ISOLATED OUTLIERS)")
    print("="*80)
    for idx in range(num_to_display):
        w_idx = sorted_by_isolation[idx]
        print(f"{idx+1:2d}. {strings[w_idx]:<18} (Distance to nearest word: {nearest_1st_dists[w_idx]:.4f})")

    print("\n" + "="*80)
    print(f"TOP {num_to_display} WORDS WITH THE LARGEST AVERAGE DISTANCE TO NEAREST {actual_k} WORDS")
    print("="*80)
    for idx in range(num_to_display):
        w_idx = sorted_by_avg_isolation[idx]
        print(f"{idx+1:2d}. {strings[w_idx]:<18} (Avg distance to nearest {actual_k}: {avg_dists[w_idx]:.4f})")

if __name__ == "__main__":
    app.run(main)
