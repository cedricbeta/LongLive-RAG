"""
KV Retrieval Bank for RAG-enhanced video generation.

Stores evicted KV cache entries at frame granularity and retrieves
the most relevant ones based on query similarity, ensuring entity
consistency across long video sequences.
"""

import torch
import torch.nn.functional as F


class KVRetrievalBank:
    """
    A per-layer retrieval bank that stores evicted KV cache frames and
    retrieves the top-k most relevant frames for the current query.

    Storage format (per frame):
        - key tokens:  [B, frame_seq_length, num_heads, head_dim]
        - value tokens: [B, frame_seq_length, num_heads, head_dim]
        - retrieval embedding: mean-pooled key across tokens [B, num_heads, head_dim]
        - global_frame_id: which frame in the video this came from

    Retrieval:
        cosine_similarity(mean_pool(current_query), stored_embeddings) → top-k frames
    """

    def __init__(self, max_frames=256, frame_seq_length=1560):
        """
        Args:
            max_frames: Maximum number of frames to store in the bank.
                        Oldest frames are dropped when exceeded.
            frame_seq_length: Number of tokens per frame (1560 for Wan 1.3B).
        """
        self.max_frames = max_frames
        self.frame_seq_length = frame_seq_length
        self.reset()

    def reset(self):
        """Clear the bank."""
        self.keys = []       # List of [B, frame_seq_length, num_heads, head_dim]
        self.values = []     # List of [B, frame_seq_length, num_heads, head_dim]
        self.embeddings = [] # List of [B, num_heads * head_dim] (flattened mean-pooled keys)
        self.frame_ids = []  # List of int (global frame index)

    def __len__(self):
        return len(self.keys)

    def store_evicted_frames(self, evicted_k, evicted_v, start_frame_id):
        """
        Store evicted KV entries into the bank, organized by frame.

        Args:
            evicted_k: [B, num_evicted_tokens, num_heads, head_dim]
            evicted_v: [B, num_evicted_tokens, num_heads, head_dim]
            start_frame_id: The global frame id of the first evicted token.
        """
        num_tokens = evicted_k.shape[1]
        num_frames = num_tokens // self.frame_seq_length

        for i in range(num_frames):
            frame_id = start_frame_id + i
            # Skip if this frame is already in the bank (e.g., from recomputation)
            if frame_id in self.frame_ids:
                continue

            start = i * self.frame_seq_length
            end = start + self.frame_seq_length
            k_frame = evicted_k[:, start:end].detach()  # [B, 1560, H, D]
            v_frame = evicted_v[:, start:end].detach()  # [B, 1560, H, D]

            # Mean-pool keys across tokens for retrieval embedding
            # [B, H, D] → [B, H*D]
            emb = k_frame.mean(dim=1).flatten(start_dim=1)  # [B, H*D]

            self.keys.append(k_frame)
            self.values.append(v_frame)
            self.embeddings.append(emb)
            self.frame_ids.append(frame_id)

        # Evict oldest if bank is full
        while len(self.keys) > self.max_frames:
            self.keys.pop(0)
            self.values.pop(0)
            self.embeddings.pop(0)
            self.frame_ids.pop(0)

    def retrieve(self, query, top_k=3, exclude_frame_ids=None):
        """
        Retrieve the top-k most relevant frames from the bank.

        Args:
            query: Current query tensor [B, num_tokens, num_heads, head_dim].
            top_k: Number of frames to retrieve.
            exclude_frame_ids: Set of frame ids to exclude (e.g., frames already
                               in the local window or sink).

        Returns:
            retrieved_k: [B, top_k * frame_seq_length, num_heads, head_dim] or None
            retrieved_v: [B, top_k * frame_seq_length, num_heads, head_dim] or None
            retrieved_frame_ids: List of retrieved frame ids, or []
        """
        if len(self.keys) == 0:
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

        # Compute query embedding: mean-pool across tokens, flatten
        # query: [B, L, H, D] → [B, H*D]
        query_emb = query.mean(dim=1).flatten(start_dim=1)  # [B, H*D]

        # Stack valid embeddings: [num_valid, B, H*D] → [B, num_valid, H*D]
        valid_embs = torch.stack(
            [self.embeddings[i] for i in valid_indices], dim=1
        )  # [B, num_valid, H*D]

        # Cosine similarity: [B, num_valid]
        sim = F.cosine_similarity(
            query_emb.unsqueeze(1),  # [B, 1, H*D]
            valid_embs,               # [B, num_valid, H*D]
            dim=-1
        )

        # Top-k (clamp to available)
        actual_k = min(top_k, len(valid_indices))
        _, topk_local_indices = sim.topk(actual_k, dim=-1)  # [B, actual_k]

        # For simplicity, use indices from batch element 0
        # (all batch elements share the same bank structure)
        topk_local = topk_local_indices[0].tolist()
        topk_bank_indices = [valid_indices[i] for i in topk_local]

        # Sort by frame_id to maintain temporal order
        topk_bank_indices.sort(key=lambda i: self.frame_ids[i])

        # Gather KV
        retrieved_k = torch.cat(
            [self.keys[i] for i in topk_bank_indices], dim=1
        )  # [B, actual_k * 1560, H, D]
        retrieved_v = torch.cat(
            [self.values[i] for i in topk_bank_indices], dim=1
        )
        retrieved_fids = [self.frame_ids[i] for i in topk_bank_indices]

        return retrieved_k, retrieved_v, retrieved_fids
