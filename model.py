import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import random
import math
import re
import os

# ==========================================
# 1. DATA PROCESSING & TOKENIZATION
# ==========================================

def tokenize(text):
    text = str(text).lower()
    # Matches words with apostrophes OR any single non-whitespace punctuation/symbol
    return re.findall(r"[\w']+|[^\w\s]", text)

class StatementDataset(Dataset):
    def __init__(self, csv_file=None):
        # Read data and topics
        if csv_file and os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            self.statements = df['statements'].tolist()
            self.topics = df['topic'].tolist()
        else:
            print(f"--> CSV '{csv_file}' not found. Using dummy data for demonstration.")
            self.statements = [
                "the quick brown fox jumps over the lazy dog.",
                "hypernetworks generate weights for other networks!",
                "pytorch makes deep learning fun and easy.",
                "autoencoders compress data into a latent space."
            ] * 20
            # Adding dummy topics for the fallback data
            self.topics = ["animals", "AI", "AI", "AI"] * 20

        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.tokenized_statements = []

        # Build Vocab
        for text in self.statements:
            tokens = tokenize(text)
            self.tokenized_statements.append(tokens)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, idx):
        tokens = self.tokenized_statements[idx]
        indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        return torch.tensor(indices, dtype=torch.long)

def collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)

# ==========================================
# 2. THE HYPERNETWORK AUTOENCODER
# ==========================================

class HyperNetworkTextAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, latent_dim=64, pos_dim=32, max_seq_len=500):
        super().__init__()
        self.vocab_size = vocab_size
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim

        # 1. ENCODER
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.to_latent = nn.Linear(hidden_dim * 2, latent_dim)

        # 2. HYPERNETWORK (Generates W1, b1 for the Classifier)
        self.gen_W1 = nn.Linear(latent_dim, hidden_dim * pos_dim)
        self.gen_b1 = nn.Linear(latent_dim, hidden_dim)

        # MEMORY OPTIMIZATION
        self.shared_decoder_proj = nn.Linear(hidden_dim, vocab_size)

        # Pre-compute Positional Encodings
        self.register_buffer("pe", self._build_positional_encoding(max_seq_len))

    def _build_positional_encoding(self, max_len):
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.pos_dim, 2).float() * (-math.log(10000.0) / self.pos_dim))
        pos_emb = torch.zeros(max_len, self.pos_dim)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        return pos_emb

    def forward(self, input_seqs):
        batch_size, seq_len = input_seqs.shape

        # --- ENCODING ---
        embedded = self.word_embedding(input_seqs)
        _, (h_n, _) = self.encoder_lstm(embedded)
        h_n_concat = torch.cat((h_n[-2], h_n[-1]), dim=-1)
        z = self.to_latent(h_n_concat)

        # --- HYPERNETWORK ---
        W1 = self.gen_W1(z).view(batch_size, self.hidden_dim, self.pos_dim)
        b1 = self.gen_b1(z).view(batch_size, self.hidden_dim, 1)

        # --- CLASSIFIER C (Decoding via Implicit Representation) ---
        pos_inputs = self.pe[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2)

        hidden = torch.bmm(W1, pos_inputs) + b1
        hidden = torch.relu(hidden)

        hidden = hidden.transpose(1, 2)
        logits = self.shared_decoder_proj(hidden)

        return logits, z

# ==========================================
# 3. TRAINING LOOP
# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    dataset = StatementDataset(csv_file='data.csv')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    print(f"Vocabulary Size: {dataset.vocab_size}")
    print(f"Total Sentences: {len(dataset)}")

    model = HyperNetworkTextAutoencoder(vocab_size=dataset.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 400
    print("\nStarting Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for input_seqs in dataloader:
            input_seqs = input_seqs.to(device)
            optimizer.zero_grad()

            logits, _ = model(input_seqs)

            logits_flat = logits.reshape(-1, dataset.vocab_size)
            targets_flat = input_seqs.reshape(-1)

            loss = criterion(logits_flat, targets_flat)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{epochs}] | Loss: {avg_loss:.4f}")

    # ==========================================
    # 4. TESTING DECODING
    # ==========================================
    print("\n--- Testing Decoding ---")
    model.eval()
    with torch.no_grad():
        sample_batch = collate_fn([dataset[0], dataset[1]]).to(device)
        test_logits, _ = model(sample_batch)
        predictions = torch.argmax(test_logits, dim=-1)

        for i in range(2):
            target_words = [dataset.reverse_vocab[idx.item()] for idx in sample_batch[i] if idx.item() != 0]
            pred_words_all = [dataset.reverse_vocab.get(idx.item(), "<UNK>") for idx in predictions[i]]
            pred_words_clean = [w for w in pred_words_all[:len(target_words)]]

            print(f"\nExample {i+1}:")
            print(f"TARGET : {' '.join(target_words)}")
            print(f"DECODED: {' '.join(pred_words_clean)}")

    # ==========================================
    # 5. TESTING LATENT Z TOPIC SIMILARITY
    # ==========================================
    print("\n--- Testing Latent Space (Topic Similarity) ---")
    for _ in range(20):
        unique_topics = list(set(dataset.topics))
        if len(unique_topics) >= 2:
            t1, t2 = unique_topics[0], unique_topics[1]

            # Grab indices for 2 sentences of topic 1, and 1 sentence of topic 2
            t1_indices = [i for i, t in enumerate(dataset.topics) if t == t1]
            t2_indices = [i for i, t in enumerate(dataset.topics) if t == t2]

            if len(t1_indices) > 2 and len(t2_indices) > 2:
                # random num
                if random.random() < 0.5:
                    idx_A, idx_B = random.sample(t2_indices, 2)
                    idx_C = random.choice(t1_indices)
                else:
                  # random indices of same topic
                  idx_A, idx_B = random.sample(t1_indices, 2)
                  idx_C = random.choice(t2_indices)

                # Print the sentences for clarity
                print(f"\n:")
                print(f"  Sentence A: {dataset.statements[idx_A]}...")
                print(f"  Sentence B: {dataset.statements[idx_B]}...")
                print(f"  Sentence C: {dataset.statements[idx_C]}...")

                # Generate z embeddings
                model.eval()
                with torch.no_grad():
                    batch = collate_fn([dataset[idx_A], dataset[idx_B], dataset[idx_C]]).to(device)
                    _, z = model(batch)

                    z_A = z[0:1] # Keep as 2D tensors for cosine similarity
                    z_B = z[1:2]
                    z_C = z[2:3]

                    # Compute Cosine Similarity (-1 to 1, higher is closer)
                    sim_same = F.cosine_similarity(z_A, z_B).item()
                    sim_diff = F.cosine_similarity(z_A, z_C).item()

                    # Compute Cosine Distance (0 to 2, lower is closer)
                    dist_same = 1.0 - sim_same
                    dist_diff = 1.0 - sim_diff

                    print("\nResults:")
                    print(f"  Same Topic Pair (A vs B)      - Cosine Similarity: {sim_same:.4f} | Cosine Distance: {dist_same:.4f}")
                    print(f"  Different Topic Pair (A vs C) - Cosine Similarity: {sim_diff:.4f} | Cosine Distance: {dist_diff:.4f}")

                    if dist_same < dist_diff:
                        print("\nSuccess: The latent representation correctly placed same-topic items closer together!")
                    else:
                        print("\nNote: Same-topic items are further apart. (This is normal since an autoencoder doesn't explicitly learn class groupings unless forced to via contrastive loss, but often semantics group naturally).")
            else:
                print("Not enough examples per topic found to run the test.")
        else:
            print("Not enough distinct topics found in the dataset to run cross-topic comparison.")

if __name__ == "__main__":
    main()