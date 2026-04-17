"""
KV Retrieval Bank for RAG-enhanced video generation.

Stores evicted KV cache entries at frame granularity and retrieves
the most relevant ones based on query similarity, ensuring entity
consistency across long video sequences.

Eviction policy: diversity-aware. When the bank is full, the most
redundant frame (highest similarity to its nearest neighbor) is
evicted first. This keeps visually distinctive frames (e.g., a
character that appeared briefly) in the bank much longer than a
naive FIFO policy would.

Embedding strategies:
    - "mean": Flat mean-pool over all tokens. Simple but background-
      dominated — most of the 1560 tokens per frame are background
      (sky, walls, ground), so entity features get diluted.
    - "norm_weighted": Weight each token by its L2 norm before pooling.
      Transformer key vectors for salient regions (people, objects,
      edges) tend to have higher norms, so this naturally emphasizes
      entity tokens over bland background. No extra parameters.
    - "salient_topk": Pool only the top-K tokens by L2 norm, ignoring
      the rest entirely. Produces a sharper entity-focused embedding
      at the cost of discarding diffuse global context.
    - "max_mean": Concatenate max-pool and mean-pool vectors. Max-pool
      captures the strongest feature at each dimension (entity peaks);
      mean-pool preserves overall scene context. 2x embedding size.
"""

import torch
import torch.nn.functional as F
import time


def _compute_embedding(tensor, strategy="norm_weighted", salient_k=256, norm_temp=0.1):
    """Compute a retrieval embedding from a [B, T, H, D] key/query tensor.

    Args:
        tensor: [B, T, num_heads, head_dim]
        strategy: One of "mean", "norm_weighted", "salient_topk", "max_mean".
        salient_k: Number of top-norm tokens to keep for "salient_topk".
        norm_temp: Temperature for softmax in "norm_weighted" (lower = sharper).

    Returns:
        embedding: [B, E] where E depends on strategy.
    """
    flat = tensor.flatten(start_dim=2)  # [B, T, H*D]

    if strategy == "mean":
        return flat.mean(dim=1)  # [B, H*D]

    elif strategy == "norm_weighted":
        # Weight tokens by their L2 norm — salient tokens (entities, edges)
        # have higher norms in transformer key spaces.
        norms = flat.norm(dim=-1, keepdim=True)  # [B, T, 1]
        weights = F.softmax(norms / norm_temp, dim=1)  # [B, T, 1]
        return (flat * weights).sum(dim=1)  # [B, H*D]

    elif strategy == "salient_topk":
        # Keep only the top-K highest-norm tokens and mean-pool those.
        norms = flat.norm(dim=-1)  # [B, T]
        k = min(salient_k, flat.shape[1])
        _, topk_idx = norms.topk(k, dim=1)  # [B, K]
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, flat.shape[2])
        topk_tokens = flat.gather(1, topk_idx_exp)  # [B, K, H*D]
        return topk_tokens.mean(dim=1)  # [B, H*D]

    elif strategy == "max_mean":
        # Concatenate max-pool (entity peaks) and mean-pool (scene context).
        max_pool = flat.max(dim=1).values  # [B, H*D]
        mean_pool = flat.mean(dim=1)       # [B, H*D]
        return torch.cat([max_pool, mean_pool], dim=-1)  # [B, 2*H*D]

    else:
        raise ValueError(f"Unknown embedding strategy: {strategy}")


class KVRetrievalBank:
    """
    A per-layer retrieval bank that stores evicted KV cache frames and
    retrieves the top-k most relevant frames for the current query.

    Storage format (per frame):
        - key tokens:  [B, frame_seq_length, num_heads, head_dim]
        - value tokens: [B, frame_seq_length, num_heads, head_dim]
        - retrieval embedding: computed from key tokens using the chosen strategy
        - global_frame_id: which frame in the video this came from

    Retrieval:
        cosine_similarity(embed(current_query), stored_embeddings) -> top-k frames
    """

    def __init__(self, max_frames=256, frame_seq_length=1560,
                 embedding_strategy="norm_weighted", salient_k=256, norm_temp=0.1,
                 random_mode=False):
        """
        Args:
            max_frames: Maximum number of frames to store in the bank.
            frame_seq_length: Number of tokens per frame (1560 for Wan 1.3B).
            embedding_strategy: "mean", "norm_weighted", "salient_topk", or "max_mean".
            salient_k: For "salient_topk": number of top-norm tokens to pool.
            norm_temp: For "norm_weighted": softmax temperature (lower = sharper).
            random_mode: If True, retrieve random frames instead of similarity-based (ablation baseline).
        """
        self.max_frames = max_frames
        self.frame_seq_length = frame_seq_length
        self.embedding_strategy = embedding_strategy
        self.salient_k = salient_k
        self.norm_temp = norm_temp
        self.random_mode = random_mode
        # Latency tracking
        self.store_time_ms = 0.0
        self.retrieve_time_ms = 0.0
        self.store_calls = 0
        self.retrieve_calls = 0
        # Retrieval log: list of (query_frame_id, retrieved_frame_ids, similarities)
        self.retrieval_log = []
        self.insertion_log = []
        # Retrieval result cache (for clean query source mode)
        self._cached_k = None
        self._cached_v = None
        self._cached_fids = []
        self.use_cache = False
        self.force_retrieve = False
        # Shared frame selection: when set, retrieve() fetches these IDs
        # instead of running similarity search (set externally by leader layer)
        self.shared_frame_ids = None
        self.reset()

    def reset(self):
        """Clear the bank."""
        self.keys = []       # List of [B, frame_seq_length, num_heads, head_dim]
        self.values = []     # List of [B, frame_seq_length, num_heads, head_dim]
        self.embeddings = [] # List of [B, num_heads * head_dim]
        self.frame_ids = []  # List of int (global frame index)
        self._cached_k = None
        self._cached_v = None
        self._cached_fids = []
        self.shared_frame_ids = None

    def __len__(self):
        return len(self.keys)

    def clear_retrieval_cache(self):
        """Clear cached retrieval results (call on prompt switch)."""
        self._cached_k = None
        self._cached_v = None
        self._cached_fids = []

    def _fetch_by_ids(self, frame_ids):
        """Fetch KV tensors for explicit frame IDs (shared index mode).

        Used when another layer has already decided which frames to retrieve.
        """
        fid_to_idx = {fid: i for i, fid in enumerate(self.frame_ids)}
        valid_ids = [fid for fid in frame_ids if fid in fid_to_idx]
        if not valid_ids:
            return None, None, []
        indices = [fid_to_idx[fid] for fid in valid_ids]
        retrieved_k = torch.cat([self.keys[i] for i in indices], dim=1)
        retrieved_v = torch.cat([self.values[i] for i in indices], dim=1)
        return retrieved_k, retrieved_v, valid_ids

    def _find_most_redundant(self):
        """
        Find the index of the most redundant frame in the bank.
        Redundancy = highest cosine similarity to its nearest neighbor.
        The most redundant frame is the one best represented by another
        frame already in the bank, so removing it loses the least info.
        """
        if len(self.embeddings) <= 1:
            return 0

        # Stack embeddings: [B, N, D]
        embs = torch.stack(self.embeddings, dim=1)
        # Use batch element 0 for efficiency: [N, D]
        embs_0 = F.normalize(embs[0], dim=-1)
        # Pairwise cosine similarity: [N, N]
        sim_matrix = embs_0 @ embs_0.T
        # Mask self-similarity with -inf
        sim_matrix.fill_diagonal_(-float('inf'))
        # For each frame, find its max similarity to any other frame
        max_sim, _ = sim_matrix.max(dim=1)  # [N]
        # The most redundant frame has the highest max_sim
        return max_sim.argmax().item()

    def store_evicted_frames(self, evicted_k, evicted_v, start_frame_id):
        """
        Store evicted KV entries into the bank, organized by frame.

        Args:
            evicted_k: [B, num_evicted_tokens, num_heads, head_dim]
            evicted_v: [B, num_evicted_tokens, num_heads, head_dim]
            start_frame_id: The global frame id of the first evicted token.
        """
        t0 = time.perf_counter()

        num_tokens = evicted_k.shape[1]
        num_frames = num_tokens // self.frame_seq_length

        for i in range(num_frames):
            frame_id = start_frame_id + i
            # Skip if this frame is already in the bank
            if frame_id in self.frame_ids:
                continue

            start = i * self.frame_seq_length
            end = start + self.frame_seq_length
            k_frame = evicted_k[:, start:end].detach()  # [B, 1560, H, D]
            v_frame = evicted_v[:, start:end].detach()  # [B, 1560, H, D]

            emb = _compute_embedding(
                k_frame, self.embedding_strategy, self.salient_k, self.norm_temp
            )

            novelty = 1.0
            if self.embeddings:
                stored_embs = torch.stack(self.embeddings, dim=1)
                emb_n = F.normalize(emb[0:1], dim=-1)
                stored_n = F.normalize(stored_embs[0], dim=-1)
                max_sim = (emb_n @ stored_n.T).max().item()
                novelty = 1.0 - max_sim
            self.insertion_log.append({
                "frame_id": frame_id,
                "novelty": round(novelty, 4),
                "embedding_norm": round(emb[0].norm().item(), 4),
            })

            self.keys.append(k_frame)
            self.values.append(v_frame)
            self.embeddings.append(emb)
            self.frame_ids.append(frame_id)

        # Diversity-aware eviction: remove the most redundant frame
        while len(self.keys) > self.max_frames:
            victim = self._find_most_redundant()
            self.keys.pop(victim)
            self.values.pop(victim)
            self.embeddings.pop(victim)
            self.frame_ids.pop(victim)

        elapsed = (time.perf_counter() - t0) * 1000
        self.store_time_ms += elapsed
        self.store_calls += 1

    def retrieve(self, query, top_k=3, exclude_frame_ids=None, query_frame_id=None):
        """
        Retrieve the top-k most relevant frames from the bank.

        Args:
            query: Current query tensor [B, num_tokens, num_heads, head_dim].
            top_k: Number of frames to retrieve.
            exclude_frame_ids: Set of frame ids to exclude (e.g., frames already
                               in the local window or sink).
            query_frame_id: The frame id being generated (for logging).

        Returns:
            retrieved_k: [B, top_k * frame_seq_length, num_heads, head_dim] or None
            retrieved_v: [B, top_k * frame_seq_length, num_heads, head_dim] or None
            retrieved_frame_ids: List of retrieved frame ids, or []
        """
        t0 = time.perf_counter()

        if len(self.keys) == 0:
            return None, None, []

        if self.random_mode:
            return self._retrieve_random(top_k, exclude_frame_ids, query_frame_id)

        if self.shared_frame_ids is not None:
            return self._fetch_by_ids(self.shared_frame_ids)

        if self.use_cache:
            if self._cached_k is not None:
                return self._cached_k, self._cached_v, self._cached_fids
            return None, None, []

        if exclude_frame_ids is None:
            exclude_frame_ids = set()

        # Filter to only frames not in exclude set
        valid_indices = [
            i for i, fid in enumerate(self.frame_ids)
            if fid not in exclude_frame_ids
        ]
        if len(valid_indices) == 0:
            return None, None, []

        # Compute query embedding using the same strategy as stored frames
        query_emb = _compute_embedding(
            query, self.embedding_strategy, self.salient_k, self.norm_temp
        )

        # Stack valid embeddings: [B, num_valid, H*D]
        valid_embs = torch.stack(
            [self.embeddings[i] for i in valid_indices], dim=1
        )

        # Cosine similarity: [B, num_valid]
        sim = F.cosine_similarity(
            query_emb.unsqueeze(1),  # [B, 1, H*D]
            valid_embs,               # [B, num_valid, H*D]
            dim=-1
        )

        # Top-k (clamp to available)
        actual_k = min(top_k, len(valid_indices))
        topk_sim, topk_local_indices = sim.topk(actual_k, dim=-1)  # [B, actual_k]

        # Use indices from batch element 0
        topk_local = topk_local_indices[0].tolist()
        topk_sims = topk_sim[0].tolist()
        topk_bank_indices = [valid_indices[i] for i in topk_local]

        # Ranking-order stats (before temporal re-sort)
        top1_score = topk_sims[0] if topk_sims else None
        margin = (topk_sims[0] - topk_sims[1]) if len(topk_sims) >= 2 else None

        # Entropy of similarity distribution over all candidates
        sim_probs = F.softmax(sim[0], dim=-1)
        entropy = -(sim_probs * (sim_probs + 1e-10).log()).sum().item()

        # Sort by frame_id to maintain temporal order
        sorted_pairs = sorted(zip(topk_bank_indices, topk_sims),
                              key=lambda x: self.frame_ids[x[0]])
        topk_bank_indices = [p[0] for p in sorted_pairs]
        topk_sims = [p[1] for p in sorted_pairs]

        # Gather KV
        retrieved_k = torch.cat(
            [self.keys[i] for i in topk_bank_indices], dim=1
        )
        retrieved_v = torch.cat(
            [self.values[i] for i in topk_bank_indices], dim=1
        )
        retrieved_fids = [self.frame_ids[i] for i in topk_bank_indices]

        # Log retrieval
        self.retrieval_log.append({
            "query_frame": query_frame_id,
            "retrieved_frames": retrieved_fids,
            "similarities": [round(s, 4) for s in topk_sims],
            "bank_size": len(self.keys),
            "top1_score": round(top1_score, 4) if top1_score is not None else None,
            "margin": round(margin, 4) if margin is not None else None,
            "entropy": round(entropy, 4),
            "num_candidates": len(valid_indices),
        })

        # Cache results for clean query source mode
        self._cached_k = retrieved_k
        self._cached_v = retrieved_v
        self._cached_fids = retrieved_fids

        elapsed = (time.perf_counter() - t0) * 1000
        self.retrieve_time_ms += elapsed
        self.retrieve_calls += 1

        return retrieved_k, retrieved_v, retrieved_fids

    def _retrieve_random(self, top_k=3, exclude_frame_ids=None, query_frame_id=None):
        """Random retrieval baseline for ablation."""
        t0 = time.perf_counter()

        if exclude_frame_ids is None:
            exclude_frame_ids = set()

        valid_indices = [
            i for i, fid in enumerate(self.frame_ids)
            if fid not in exclude_frame_ids
        ]
        if len(valid_indices) == 0:
            return None, None, []

        actual_k = min(top_k, len(valid_indices))
        perm = torch.randperm(len(valid_indices))[:actual_k].tolist()
        selected = sorted([valid_indices[p] for p in perm],
                          key=lambda i: self.frame_ids[i])

        retrieved_k = torch.cat([self.keys[i] for i in selected], dim=1)
        retrieved_v = torch.cat([self.values[i] for i in selected], dim=1)
        retrieved_fids = [self.frame_ids[i] for i in selected]

        self.retrieval_log.append({
            "query_frame": query_frame_id,
            "retrieved_frames": retrieved_fids,
            "similarities": [],
            "bank_size": len(self.keys),
            "mode": "random",
        })

        elapsed = (time.perf_counter() - t0) * 1000
        self.retrieve_time_ms += elapsed
        self.retrieve_calls += 1

        return retrieved_k, retrieved_v, retrieved_fids
