import json
import torch
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

HF_BACKBONE = "avsolatorio/GIST-Embedding-v0"
hf_tokenizer = AutoTokenizer.from_pretrained(HF_BACKBONE)
hf_model = AutoModel.from_pretrained(HF_BACKBONE).to(device)

# ST model for fallback + query encoding
st_model = SentenceTransformer(HF_BACKBONE, device=device)


# ---------------------------------------------------
# ① Encode document into HF token embeddings (long-doc safe)
# ---------------------------------------------------
def encode_full_doc_tokens(text, max_len=512):
    """
    Encode long document safely by splitting into <=512-token segments.
    ALWAYS returns 2D tensors to avoid HF shape errors.
    """
    # First tokenize WITHOUT tensors so we can manually segment
    base = hf_tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    ids = base["input_ids"]
    offsets = base["offset_mapping"]

    all_emb = []
    all_offsets = []

    # Split long text into segments of <=512 tokens
    for i in range(0, len(ids), max_len):
        seg_ids = ids[i:i+max_len]
        seg_offsets = offsets[i:i+max_len]

        # ⚠️ IMPORTANT: wrap inside batch dimension manually
        seg_inputs = {
            "input_ids": torch.tensor([seg_ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, len(seg_ids)), dtype=torch.long, device=device)
        }

        with torch.no_grad():
            out = hf_model(**seg_inputs)

        hidden = out.last_hidden_state.squeeze(0)   # shape = (seg_len, hidden_dim)
        all_emb.append(hidden.cpu())
        all_offsets.extend(seg_offsets)

    full_emb = torch.cat(all_emb, dim=0).to(device)
    return full_emb, all_offsets



# ---------------------------------------------------
# ② Map chunk text to token span
# ---------------------------------------------------
def find_token_span_for_chunk(chunk_text, full_text, offsets):
    start_char = full_text.find(chunk_text)
    if start_char == -1:
        return None, None

    end_char = start_char + len(chunk_text)

    start_tok = None
    end_tok = None

    for i, (c_start, c_end) in enumerate(offsets):
        if c_start <= start_char < c_end:
            start_tok = i
        if c_start < end_char <= c_end:
            end_tok = i

    if start_tok is None:
        return None, None
    if end_tok is None:
        end_tok = start_tok

    return start_tok, end_tok


# ---------------------------------------------------
# ③ Pooling
# ---------------------------------------------------
def pool_chunk(token_emb, start_tok, end_tok):
    sub = token_emb[start_tok:end_tok+1]
    return sub.mean(dim=0)


# ---------------------------------------------------
# ④ Late Chunking Encode API
# ---------------------------------------------------
def late_chunking_encode(html_text, chunks):
    soup = BeautifulSoup(html_text, "html.parser")
    full_text = soup.get_text(" ", strip=True)

    token_emb, offsets = encode_full_doc_tokens(full_text)

    out = []

    for ch in chunks:
        st, ed = find_token_span_for_chunk(ch, full_text, offsets)
        if st is None:
            # fallback
            out.append(st_model.encode(ch, convert_to_tensor=True))
            continue

        pooled = pool_chunk(token_emb, st, ed)
        out.append(pooled)

    return torch.stack(out).to(device)


# ---------------------------------------------------
# ⑤ Simple sliding window chunker
# ---------------------------------------------------
def sliding_window_chunk(text, window=128, overlap=32):
    words = text.split()
    step = window - overlap
    out = []
    i = 0
    while i < len(words):
        out.append(" ".join(words[i:i+window]))
        i += step
    return out


# ---------------------------------------------------
# ⑥ Find gold chunk using your document_tokens
# ---------------------------------------------------
def find_gold_chunk(chunks, document_tokens, start_tok, end_tok):
    # convert official tokens to string
    gold = " ".join([document_tokens[i]["token"].lower()
                     for i in range(start_tok, end_tok)
                     if i < len(document_tokens)])
    if not gold:
        return None

    # exact search
    for i, ch in enumerate(chunks):
        if gold in ch.lower():
            return i
    return None


# ---------------------------------------------------
# ⑦ Evaluation
# ---------------------------------------------------
def evaluate(dataset_path):
    ranks = []
    skipped = 0

    with open(dataset_path, "r", encoding="utf-8") as f:
        for item in tqdm(map(json.loads, f), desc="Evaluating Late Chunking"):

            html = item["document_html"]
            question = item["question_text"]
            doc_tokens = item["document_tokens"]

            chunks = sliding_window_chunk(
                BeautifulSoup(html, "html.parser").get_text(" ", strip=True),
                window=128, overlap=32
            )
            if not chunks:
                skipped += 1
                continue

            chunk_emb = late_chunking_encode(html, chunks)
            q_emb = st_model.encode(question, convert_to_tensor=True)

            scores = util.cos_sim(q_emb, chunk_emb)[0]
            ranking = scores.argsort(descending=True).cpu().numpy()

            ann = item["annotations"][0]
            if ann["short_answers"]:
                gs = ann["short_answers"][0]["start_token"]
                ge = ann["short_answers"][0]["end_token"]
            else:
                gs = ann["long_answer"]["start_token"]
                ge = ann["long_answer"]["end_token"]

            if gs < 0 or ge < 0:
                skipped += 1
                continue

            gold_idx = find_gold_chunk(chunks, doc_tokens, gs, ge)
            if gold_idx is None:
                skipped += 1
                continue

            rank = np.where(ranking == gold_idx)[0][0] + 1
            ranks.append(rank)

    recall10 = np.mean([1 if r <= 10 else 0 for r in ranks]) if ranks else 0
    mrr = np.mean([1.0 / r for r in ranks]) if ranks else 0
    print("Recall@10:", recall10, "MRR:", mrr)


# ---------------------------------------------------
# ⑧ Run
# ---------------------------------------------------
if __name__ == "__main__":
    evaluate("data/nq_filtered_short.jsonl")
