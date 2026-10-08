"""
RAGenome Transformer.

Input:

    x = [ species_0 | species_1 | ... | species_{N-1} | query ]

Every species carries a taxonomy embedding (added to each of its tokens) and
RoPE positions given by the original alignment column of each retrieved token.

    * Intra-species local attention runs independently inside every sequence.
    * Inter-species cross-attention lets the query attend to the whole retrieved array.

The layers are arranged in two nested levels (an outer and an inner `Level`), each with `local_depth`
intra-species local-attention layers before and after its valley, so `local_depth = 3` and
`cross_depth = 3` give 6 local-attention layers, 3 inter-species cross-attention layers and
6 local-attention layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ragenome.models.nn.layers import CrossAttention, Level, Packed


class RAGenomeTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        local_depth: int,
        cross_depth: int,
        sliding_window: int,
        heads: int,
        dim_head: int,
        rotary_emb_dim: int,
        seq_vocab_size: int = 11,  # 7 special tokens + a, c, g, t
        retrieved_taxonomy_vocab_size: int = 317,  # distinct NCBI taxids
    ):
        super().__init__()
        assert dim % 8 == 0, "dim must be divisible by 8 for the taxonomy embedding"

        self.seq_embedding_layer = nn.Embedding(seq_vocab_size, dim)
        # Taxonomy embedding: 8 NCBI lineage levels x (dim // 8) = dim, the concatenation of the per-level embeddings
        # fills the model dimension without any extra projection.
        self.retrieved_taxonomy_embedding = nn.Embedding(retrieved_taxonomy_vocab_size, dim // 8)

        attn_kwargs = dict(
            sliding_window=sliding_window, heads=heads, dim_head=dim_head,
            rotary_emb_dim=rotary_emb_dim,
        )
        cross_attention = CrossAttention(dim, heads, dim_head, cross_depth, rotary_emb_dim)
        inner = Level(dim, local_depth, cross_attention, norm_out=False, **attn_kwargs)
        self.core = Level(dim, local_depth, inner, norm_out=True, **attn_kwargs)
        self.seq_prediction_head = nn.Linear(dim, seq_vocab_size)

    def forward(
        self,
        genome_input_ids: torch.Tensor,
        packed_retrieved_ids: torch.Tensor,
        retrieved_lengths: torch.Tensor,
        packed_aligned_pos: torch.Tensor,
        retrieved_taxonomy: torch.Tensor,
        query_taxonomy: torch.Tensor,
    ) -> torch.Tensor:
        
        B, L = genome_input_ids.shape
        n_ret = packed_retrieved_ids.shape[1]
        N = retrieved_lengths.shape[1]
        device = genome_input_ids.device

        positions = torch.arange(n_ret, device=device).expand(B, -1).contiguous()
        species_ids = torch.searchsorted(retrieved_lengths.cumsum(dim=1), positions, right=True)
        species_ids = species_ids.masked_fill(species_ids == N, -1)

        # Retrieved tokens: nucleotide embedding + the taxonomy embedding of the token's species.
        x_retrieved = self.seq_embedding_layer(packed_retrieved_ids)
        D = x_retrieved.shape[-1]
        species_emb = self.retrieved_taxonomy_embedding(retrieved_taxonomy).reshape(B, N, D)
        tax_per_token = species_emb.gather(
            dim=1, index=species_ids.clamp(min=0).unsqueeze(-1).expand(-1, -1, D)
        )  # padding positions -> species 0, zeroed below
        x_retrieved = x_retrieved + tax_per_token * (species_ids >= 0).unsqueeze(-1)

        # Query: nucleotide embedding + the taxonomy embedding of the query species.
        x_query = self.seq_embedding_layer(genome_input_ids)
        x_query = x_query + self.retrieved_taxonomy_embedding(query_taxonomy).reshape(B, 1, -1)

        # Packed layout [species_0 | ... | species_{N-1} | query]; the query is sequence N.
        query_ids = torch.full((B, L), N, dtype=torch.long, device=device)
        query_pos = torch.arange(L, device=device).expand(B, -1)
        all_lens = torch.cat(
            [retrieved_lengths, torch.full((B, 1), L, dtype=torch.long, device=device)], dim=1
        )  # (B, N+1)
        packed = Packed(
            # RoPE positions: alignment columns for retrieved tokens, local index for the query
            pos_ids=torch.cat([packed_aligned_pos, query_pos], dim=1),
            doc_ids=torch.cat([species_ids, query_ids], dim=1),
            cu_seqlens=F.pad(all_lens.reshape(-1).cumsum(0), (1, 0)).to(torch.int32),
            max_seqlen=int(all_lens.max()),
            n_retrieved=n_ret,
        )

        x = self.core(torch.cat([x_retrieved, x_query], dim=1), packed)

        # Extract the query representation and predict.
        return self.seq_prediction_head(x[:, n_ret:])  # (B, L, seq_vocab_size)
