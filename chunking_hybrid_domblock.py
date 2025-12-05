import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from bs4 import BeautifulSoup
import re
from typing import List, Tuple, Dict, Optional, Any, cast
import bs4
from collections import defaultdict

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


# Parameters
WINDOW = 512
OVERLAP = 102
SEMANTIC_THRESHOLD = 0.5

# Dataset
DATASET_PATH = "data/nq_filtered_medium.jsonl"

# Model
MODEL_NAME = "avsolatorio/GIST-Embedding-v0"

# Methods to compare
CHUNKING_STRATEGIES = ["sliding_window", "html_aware", "semantic_similarity", "hybrid_html_semantic", "dom_block_tree"]

print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device=device)

print("Model loaded successfully!")
print(f"Max sequence length: {model.max_seq_length}")

# 3 Chunking Strategies
def sliding_window_chunk(text, window=512, overlap=102):
    """Fixed-size sliding window chunking."""
    words = text.split()
    chunks = []
    step = window - overlap
    
    i = 0
    while i < len(words):
        chunk_words = words[i:i + window]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        i += step
    
    return chunks


def html_aware_chunk(html_text, max_chunk_size=512):
    """HTML-structure-aware chunking."""
    soup = BeautifulSoup(html_text, 'html.parser')
    chunks = []
    structural_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'td', 'div']
    
    current_chunk = []
    current_word_count = 0
    
    def add_chunk():
        if current_chunk:
            chunks.append(" ".join(current_chunk))
    
    for element in soup.find_all(structural_tags):
        text = element.get_text(strip=True)
        if not text:
            continue
        
        words = text.split()
        
        if current_word_count + len(words) > max_chunk_size and current_chunk:
            add_chunk()
            current_chunk = []
            current_word_count = 0
        
        current_chunk.extend(words)
        current_word_count += len(words)
        
        if element.name in ['h1', 'h2', 'h3'] and current_word_count > max_chunk_size * 0.5:
            add_chunk()
            current_chunk = []
            current_word_count = 0
    
    add_chunk()
    
    # Fallback
    if not chunks:
        words = html_text.split()
        for i in range(0, len(words), max_chunk_size):
            chunk_words = words[i:i + max_chunk_size]
            chunks.append(" ".join(chunk_words))
    
    return chunks


def semantic_similarity_chunk(text, model, threshold=0.5, max_chunk_size=512):
    """Semantic similarity-based chunking."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 1:
        return [text]
    
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
    
    similarities = []
    for i in range(len(sentence_embeddings) - 1):
        sim = util.cos_sim(sentence_embeddings[i], sentence_embeddings[i + 1])[0][0].item()
        similarities.append(sim)
    
    chunks = []
    current_chunk = [sentences[0]]
    current_word_count = len(sentences[0].split())
    
    for i, sim in enumerate(similarities):
        next_sentence = sentences[i + 1]
        next_word_count = len(next_sentence.split())
        
        if sim < threshold or (current_word_count + next_word_count > max_chunk_size):
            chunks.append(" ".join(current_chunk))
            current_chunk = [next_sentence]
            current_word_count = next_word_count
        else:
            current_chunk.append(next_sentence)
            current_word_count += next_word_count
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Extension1: Hybrid HTML-Semantic Chunking
def hybrid_html_semantic_chunk(html_text, model, threshold=0.5, max_chunk_size=512):
    """
    Hybrid chunking:
    1. Use HTML heading hierarchy (<h1>-<h6>) to segment the document into sections.
    2. Inside each section, apply semantic similarity-based chunking.
    Returns a list of final chunks (strings).
    """
    soup = BeautifulSoup(html_text, 'html.parser')

    # Step 1: collect sections by heading structure
    sections = []
    current_section = []

    for element in soup.find_all(['h1','h2','h3','h4','h5','h6','p','li','td','div']):
        text = element.get_text(strip=True)
        if not text:
            continue

        # Start a new section on any heading
        if element.name in ['h1','h2','h3','h4','h5','h6']:
            if current_section:
                sections.append(" ".join(current_section))
                current_section = []
        current_section.append(text)

    if current_section:
        sections.append(" ".join(current_section))

    # Step 2: apply semantic chunking inside each section
    final_chunks = []

    for section in sections:
        sentences = re.split(r'(?<=[.!?])\s+', section)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            # Short section directly appended
            final_chunks.append(section)
            continue
        
        # Embed sentences
        embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        
        # Compute similarities between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = util.cos_sim(embeddings[i], embeddings[i+1])[0][0].item()
            similarities.append(sim)
        
        # Build chunks within the section
        current_chunk = [sentences[0]]
        current_len = len(sentences[0].split())

        for i, sim in enumerate(similarities):
            next_sentence = sentences[i+1]
            next_len = len(next_sentence.split())

            # boundary: semantic break OR token overflow
            if sim < threshold or current_len + next_len > max_chunk_size:
                final_chunks.append(" ".join(current_chunk))
                current_chunk = [next_sentence]
                current_len = next_len
            else:
                current_chunk.append(next_sentence)
                current_len += next_len

        # Add the last chunk inside section
        if current_chunk:
            final_chunks.append(" ".join(current_chunk))

    return final_chunks

#Extension2: DOM Block Tree Chunking

def build_block_tree(html: str, max_node_words: int=512, zh_char=False) -> Tuple[List[Tuple[bs4.element.Tag, List[str], bool]], str]:
    soup = bs4.BeautifulSoup(html, 'html.parser')
    word_count = len(soup.get_text()) if zh_char else len(soup.get_text().split())
    if word_count > max_node_words:
        possible_trees = [(soup, [])]
        target_trees = []  # [(tag, path, is_leaf)]
        #  split the entire dom tee into subtrees, until the length of the subtree is less than max_node_words words
        #  find all possible trees
        while True:
            if len(possible_trees) == 0:
                break
            tree = possible_trees.pop(0)
            tag_children = defaultdict(int)
            bare_word_count = 0
            #  count child tags
            for child in tree[0].contents:
                if isinstance(child, bs4.element.Tag):
                    tag_children[child.name] += 1
            _tag_children = {k: 0 for k in tag_children.keys()}

            #  check if the tree can be split
            for child in tree[0].contents:
                if isinstance(child, bs4.element.Tag):
                    #  change child tag with duplicate names
                    if tag_children[child.name] > 1:
                        new_name = f"{child.name}{_tag_children[child.name]}"
                        new_tree = (child, tree[1] + [new_name])
                        _tag_children[child.name] += 1
                        child.name = new_name
                    else:
                        new_tree = (child, tree[1] + [child.name])
                    word_count = len(child.get_text()) if zh_char else len(child.get_text().split())
                    #  add node with more than max_node_words words, and recursion depth is less than 64
                    if word_count > max_node_words and len(new_tree[1]) < 64:
                        possible_trees.append(new_tree)
                    else:
                        target_trees.append((new_tree[0], new_tree[1], True))
                else:
                    bare_word_count += len(str(child)) if zh_char else len(str(child).split())

            #  add leaf node
            if len(tag_children) == 0:
                target_trees.append((tree[0], tree[1], True))
            #  add node with more than max_node_words bare words
            elif bare_word_count > max_node_words:
                target_trees.append((tree[0], tree[1], False))
    else:
        soup_children = [c for c in soup.contents if isinstance(c, bs4.element.Tag)]
        if len(soup_children) == 1:
            target_trees = [(soup_children[0], [soup_children[0].name], True)]
        else:
            # add an html tag to wrap all children
            new_soup = bs4.BeautifulSoup("", 'html.parser')
            new_tag = new_soup.new_tag("html")
            new_soup.append(new_tag)
            for child in soup_children:
                new_tag.append(child)
            target_trees = [(new_tag, ["html"], True)]

    html=str(soup)
    return target_trees, html

def dom_block_tree_chunk(html_text, max_node_words=512):
    block_tree, _ = build_block_tree(html_text, max_node_words=max_node_words)
    chunks = [tag.get_text(" ", strip=True) for tag, path, is_leaf in block_tree]
    return chunks

# Helper Functions
def find_gold_chunk_by_content(chunks, document_tokens, start_token, end_token, overlap_threshold=0.3):
    """Find which chunk contains the gold answer.
    - Uses exact substring match first
    - Falls back to relaxed word-overlap threshold (default 30%)
    """
    try:
        if start_token is None or end_token is None:
            return None
        if start_token < 0 or end_token < 0 or start_token >= len(document_tokens):
            return None
    except Exception:
        return None

    gold_tokens = []
    for i in range(start_token, min(end_token, len(document_tokens))):
        tok = document_tokens[i]
        # Some datasets may store tokens differently
        if isinstance(tok, dict):
            gold_tokens.append(tok.get('token', ''))
        else:
            gold_tokens.append(str(tok))

    gold_text = " ".join([t for t in gold_tokens if t]).strip().lower()
    if not gold_text:
        return None

    # Exact match
    for idx, chunk in enumerate(chunks):
        if not chunk:
            continue
        if gold_text in chunk.lower():
            return idx

    # Fuzzy match (relaxed)
    gold_words = set(gold_text.split())
    if not gold_words:
        return None
    
    # ----------- DYNAMIC THRESHOLD ADJUSTMENT HERE -----------
    n = len(gold_words)

    if n == 1:
        # For 1-token answers, disable fuzzy matching entirely
        return None

    elif n == 2:
        # Must match both tokens
        dynamic_threshold = 1.0

    elif n == 3:
        # Must match at least 2 tokens (≈0.66)
        dynamic_threshold = 0.8

    else:
        # Use original threshold
        dynamic_threshold = overlap_threshold
    # ----------------------------------------------------------

    best_idx = None
    best_overlap = 0
    for idx, chunk in enumerate(chunks):
        if not chunk:
            continue
        chunk_words = set(chunk.lower().split())
        overlap = len(gold_words & chunk_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    # # Require a minimum fraction of gold words to appear in the chunk
    # if gold_words and (best_overlap / max(len(gold_words), 1)) >= overlap_threshold:
    #     return best_idx
    
    # Apply dynamic threshold
    if best_idx is not None and (best_overlap / max(n, 1)) >= dynamic_threshold:
        return best_idx

    return None


def compute_metrics(rank_list):
    """Compute Recall@10 and MRR."""
    recall10 = np.mean([1 if r <= 10 else 0 for r in rank_list])
    mrr = np.mean([1.0 / r for r in rank_list])
    return recall10, mrr

def compute_ndcg(rank_list):
    """Compute overall nDCG for single relevant item."""
    vals = []
    for r in rank_list:
        ndcg = 1.0 / np.log2(r + 1)
        vals.append(ndcg)
    return float(np.mean(vals)) if vals else 0.0


# Evaluation Loop
def evaluate_method(model, dataset_path, chunking_strategy, method_name, **chunk_params):
    """
    Evaluate a chunking strategy with naive encoding.
    
    Args:
        model: SentenceTransformer model
        dataset_path: Path to dataset
        chunking_strategy: Function that returns chunk texts
        method_name: Name for logging
        **chunk_params: Parameters for chunking function
    
    Returns:
        Dictionary with metrics
    """
    rank_list = []
    skipped = 0
    skip_reasons = {
        'parse_error': 0,
        'chunking_error': 0,
        'no_chunks': 0,
        'encode_error': 0,
        'no_annotations': 0,
        'invalid_gold_tokens': 0,
        'gold_chunk_not_found': 0,
    }
    num_chunks_list = []
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Evaluating {method_name}"):
            try:
                item = json.loads(line)
            except Exception:
                skipped += 1
                skip_reasons['parse_error'] += 1
                continue
            
            question = item["question_text"]
            html_text = item["document_html"]
            doc_tokens = item["document_tokens"]
            
            # Apply chunking strategy
            try:
                if "semantic" in method_name:
                    chunks = chunking_strategy(html_text, model, **chunk_params)
                else:
                    chunks = chunking_strategy(html_text, **chunk_params)
            except Exception:
                skipped += 1
                skip_reasons['chunking_error'] += 1
                continue
            
            if not chunks:
                skipped += 1
                skip_reasons['no_chunks'] += 1
                continue
            else:
                num_chunks_list.append(len(chunks))
            
            # Encode chunks (naive encoding: each chunk independently)
            try:
                chunk_embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
            except Exception:
                skipped += 1
                skip_reasons['encode_error'] += 1
                continue
            
            # Encode query
            query_embedding = model.encode(question, convert_to_tensor=True, show_progress_bar=False)
            
            # Similarity ranking
            scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
            ranking = scores.argsort(descending=True).cpu().numpy()
            
            # Find gold answer
            annotations = item.get("annotations", [])
            if not annotations:
                skipped += 1
                skip_reasons['no_annotations'] += 1
                continue

            ann = annotations[0]
            if ann.get("short_answers"):
                gold_start = ann["short_answers"][0].get("start_token", -1)
                gold_end = ann["short_answers"][0].get("end_token", -1)
            else:
                la = ann.get("long_answer", {})
                gold_start = la.get("start_token", -1)
                gold_end = la.get("end_token", -1)
            
            if gold_start is None or gold_end is None or gold_start < 0 or gold_end < 0:
                skipped += 1
                skip_reasons['invalid_gold_tokens'] += 1
                continue
            
            gold_chunk = find_gold_chunk_by_content(chunks, doc_tokens, gold_start, gold_end)
            if gold_chunk is None or gold_chunk >= len(chunks):
                skipped += 1
                skip_reasons['gold_chunk_not_found'] += 1
                continue
            
            # Find rank
            gold_rank = np.where(ranking == gold_chunk)[0][0] + 1
            rank_list.append(gold_rank)
            # print(f"Gold rank: {gold_rank}")
            # break
    
    # Compute metrics
    if rank_list:
        recall10, mrr = compute_metrics(rank_list)
    else:
        recall10, mrr = 0.0, 0.0
    ndcg = compute_ndcg(rank_list)

    avg_num_chunks = float(np.mean(num_chunks_list)) if num_chunks_list else 0.0
    
    return {
        "method": method_name,
        "recall@10": recall10,
        "mrr": mrr,
        "ndcg": ndcg,
        "total_samples": len(rank_list),
        "skipped": skipped,
        # "skip_reasons": skip_reasons,
        "avg_num_chunks": avg_num_chunks
    }


results = {}

print("="*70)
print("Running Chunking Strategy Comparison")
print("="*70)

strategies_map = {
    # "sliding_window": (sliding_window_chunk, {"window": WINDOW, "overlap": OVERLAP}),
    "html_aware": (html_aware_chunk, {"max_chunk_size": WINDOW}),
    # "semantic_similarity": (semantic_similarity_chunk, {"threshold": SEMANTIC_THRESHOLD, "max_chunk_size": WINDOW}),
    # "hybrid_html_semantic": (hybrid_html_semantic_chunk, {"threshold": SEMANTIC_THRESHOLD, "max_chunk_size": WINDOW}),
    "dom_block_tree": (dom_block_tree_chunk, {"max_node_words": WINDOW}),
}

for strategy_name, (strategy_func, params) in strategies_map.items():
    print(f"\n{'='*70}")
    print(f"Chunking Strategy: {strategy_name.upper()}")
    print(f"{'='*70}")
    
    results[strategy_name] = evaluate_method(
        model=model,
        dataset_path=DATASET_PATH,
        chunking_strategy=strategy_func,
        method_name=strategy_name,
        **params
    )
    print(
        f"  Recall@10: {results[strategy_name]['recall@10']:.4f}, "
        f"MRR: {results[strategy_name]['mrr']:.4f}, "
        f"nDCG: {results[strategy_name].get('ndcg', 0.0):.4f}, "
        f"Avg #chunks: {results[strategy_name].get('avg_num_chunks', 0.0):.2f}"
    )

print("\n" + "="*70)
print("All experiments completed!")
print("="*70)


# Create results DataFrame
df_results = pd.DataFrame(results).T

print("\n=== Chunking Strategy Comparison Results ===")
print(df_results.to_string())

# Save to CSV
df_results.to_csv("chunking_comparison_results.csv")
print("\nResults saved to chunking_comparison_results.csv")


print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)

# Sort by Recall@10
sorted_by_recall = sorted(results.items(), key=lambda x: x[1]['recall@10'], reverse=True)

print("\nRanking by Recall@10:")
for i, (strategy, metrics) in enumerate(sorted_by_recall, 1):
    print(f"  {i}. {strategy.upper().replace('_', ' ')}: Recall@10={metrics['recall@10']:.4f}, MRR={metrics['mrr']:.4f}")

# Best vs Worst comparison
best_strategy, best_metrics = sorted_by_recall[0]
worst_strategy, worst_metrics = sorted_by_recall[-1]

recall_improvement = ((best_metrics['recall@10'] - worst_metrics['recall@10']) / worst_metrics['recall@10'] * 100) if worst_metrics['recall@10'] > 0 else 0
mrr_improvement = ((best_metrics['mrr'] - worst_metrics['mrr']) / worst_metrics['mrr'] * 100) if worst_metrics['mrr'] > 0 else 0

print(f"\n🏆 Best Strategy: {best_strategy.upper().replace('_', ' ')}")
print(f"   Recall@10: {best_metrics['recall@10']:.4f}")
print(f"   MRR: {best_metrics['mrr']:.4f}")

print(f"\n📊 Improvement over worst strategy:")
print(f"   Recall@10: {recall_improvement:+.2f}%")
print(f"   MRR: {mrr_improvement:+.2f}%")