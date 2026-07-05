# ACN Plot

A web application that takes in a word and outputs its $n$ nearest phonetic neighbors as an interactive 2D edge-weighted graph. It uses Apple's [Acoustic Neighbor Embeddings](https://github.com/apple/ml-acn-embed) to compute similarities and t-SNE for the 2D layout.

## Setup

### 1. Clone the repository
Because this project depends on `ml-acn-embed` as a Git submodule, you need to make sure the submodule is initialized when you clone the project:

```bash
git clone --recursive git@github.com:kentslaney/acn-plot.git
cd acn-plot
```

*If you already cloned the repository without the `--recursive` flag, you can fetch the submodule by running:*
```bash
git submodule update --init --recursive
```

### 2. Set up the virtual environment
Create a virtual environment and install the required dependencies (including the `ml-acn-embed` package in editable mode):

```bash
python3 -m venv venv
source venv/bin/activate
pip install flask scikit-learn numpy torch torchaudio
pip install -e ml-acn-embed
```

### 3. Download the models and data
Download the pretrained acoustic embedding models and the pruned wakeword dictionary:

```bash
curl -s https://ml-site.cdn-apple.com/models/ml-acn-embed/model.tgz | tar xz
curl -s https://ml-site.cdn-apple.com/models/ml-acn-embed/wakeword.tgz | tar xz
```

### 4. Run the webserver
Start the Flask application (it runs on port 5001 to avoid conflicts with macOS AirPlay Receiver):

```bash
python app.py
```

Open your browser and navigate to [http://127.0.0.1:5001](http://127.0.0.1:5001) to interact with the graph!
