import torch.nn as nn
import torch
from einops import repeat
import math
from mzsane.models.def_transformer import TransformerEncoder
import logging


class AE(nn.Module):
    def __init__(self, config):
        # TODO
        super(AE, self).__init__()
        # instanciate components
        i_dim = config.get("ae:i_dim", 201)
        d_model = config.get("ae:d_model", 512)
        nhead = config.get("ae:nhead", 8)
        num_layers = config.get("ae:num_layers", 6)
        lat_dim = config.get("ae:lat_dim", 16)
        windowsize = config.get("training::windowsize", 16)
        dropout = config.get("ae:dropout", 0.0)

        assert (
            d_model % nhead == 0
        ), f"invalid transformer config with d_model {d_model} and n_heads {nhead}"

        # mapping to token_dim
        self.tokenizer = nn.Linear(i_dim, d_model)
        # encoder
        if config.get("ae:transformer_type", "pytorch") == "pytorch":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
        elif config.get("ae:transformer_type", "pytorch") == "gpt2":
            self.transformer_encoder = TransformerEncoder(
                n_layer=num_layers,
                n_head=nhead,
                d_model=d_model,
                dropout=dropout,
                bias=False,
                causal=False,
                block_size=windowsize,
            )
        else:
            raise ValueError(
                f"invalid encoder type {config.get('ae:transformer_type')}"
            )
        # mapping from token_dim to lat_dim
        self.encoder_comp = nn.Linear(d_model, lat_dim)

        # decoder
        # mapping from token_dim to original dim
        self.detokenizer = nn.Linear(d_model, i_dim)

        # decoder is built of __ENcoder__ layers
        if config.get("ae:transformer_type", "pytorch") == "pytorch":
            decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
            self.transformer_decoder = nn.TransformerEncoder(
                decoder_layer, num_layers=num_layers
            )
        elif config.get("ae:transformer_type", "pytorch") == "gpt2":
            self.transformer_decoder = TransformerEncoder(
                n_layer=num_layers,
                n_head=nhead,
                d_model=d_model,
                dropout=dropout,
                bias=False,
                causal=False,
                block_size=windowsize,
            )
        else:
            raise ValueError(
                f"invalid encoder type {config.get('ae:transformer_type')}"
            )
        # mapping from lat_dim to token_dim
        self.decoder_comp = nn.Linear(lat_dim, d_model)

        # position encoder
        max_positions = config.get("ae:max_positions", [48, d_model])
        self.pe = PositionEmbs(max_positions=max_positions, embedding_dim=d_model)

        # projection head
        self.projection_head = SimpleProjectionHead(
            d_model=lat_dim, n_tokens=windowsize, odim=30
        )

        # dropout
        self.dropout = nn.Dropout(dropout)

        # taken from Kaparthy's GPT2 implementation:
        init_type = config.get("ae:weight_init_type", "normal")
        self.initialize_weights(init_type=init_type)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

        # aggregate embeddings
        self.aggregate_embeddings = config.get("ae:aggregate_embeddings", "mean")

    def initialize_weights(self, init_type="normal"):
        """
        Apply the weight initialization to all submodules using the given init_type.
        """
        def init_fn(module):
            self._init_weights(module, init_type=init_type)
        
        self.apply(init_fn)


    def _init_weights(self, module, init_type="normal"):
        """
        Initializes weights of the model based on module type and init_type
        Args:
            module: nn.Module
            init_type: str
        """
        if isinstance(module, torch.nn.Linear):
            if init_type == "normal":
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(module.weight)
            else:
                raise ValueError(f"invalid init type {init_type}")
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, torch.nn.Embedding):
            # Embeddings are often initialized with normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, torch.nn.LayerNorm):
            # LayerNorm layers typically use constant initialization
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    #
    def forward(self, x: torch.tensor, p: torch.tensor, mask=None, return_z_stats=False):
        """
        passes sequence of embeddings through encoder / decoder transformer
        Args:
            x: torch.tensor sequence of weight/channel tokens
            p: torch.tensor sequence of positions
            mask: optional torch.tensor mask for attention
        Returns:
            z: torch.tensor sequence of latent representations
            y: torch.tensor sequence of reconstructions
        """
        z = self.forward_encoder(x, p, mask)
        zp = self.projection_head(z)
        if return_z_stats:
            y, z_out_dict = self.forward_decoder(z, p, mask=mask, return_z_stats=return_z_stats)
            return z, y, zp, z_out_dict
        else:
            y = self.forward_decoder(z, p, mask=mask)
            return z, y, zp

    def forward_encoder(
        self, x: torch.tensor, p: torch.tensor, mask=None
    ) -> torch.tensor:
        """
        Args:
            x: torch.tensor sequence of weight/channel tokens
            p: torch.tensor sequence of positions
            mask: optional torch.tensor mask for attention
        Returns:
            z: torch.tensor sequence of latent representations
        """
        # map weight tokens from input dim to d_model
        x = self.tokenizer(x)
        # add position embeddings
        x = self.pe(x, p)
        # apply dropout
        x = self.dropout(x)
        # pass through encoder transformer
        x = self.transformer_encoder(x, mask=mask)
        # compress to latent dim
        x = self.encoder_comp(x)
        # return
        return x

    def forward_decoder(
        self, z: torch.tensor, p: torch.tensor, mask=None, return_z_stats=False
    ) -> torch.tensor:
        """
        Args:
            z: torch.tensor sequence of latent representations
            p: torch.tensor sequence of positions
            mask: optional torch.tensor mask for attention
        Returns:
            y: torch.tensor sequence of reconstructions
        """
        # map weight tokens from latent dim to d_model
        z = self.decoder_comp(z)
        # add position embeddings (again)
        z = self.pe(z, p)
        # apply dropout
        z = self.dropout(z)
        # pass through decoder transformer
        z = self.transformer_decoder(z, mask=mask)
        # map back to original dim (so that it can be cast to checkpoint)
        if return_z_stats:
            z_norm_before_detokenizer = torch.linalg.norm(z.view(z.shape[0], -1), ord=2, dim=1).mean()
            z_var_before_detokenizer = torch.mean(torch.var(z.view(z.shape[0], -1), dim=0))
        z = self.detokenizer(z)
        if return_z_stats:
            z_norm_after_detokenizer = torch.linalg.norm(z.view(z.shape[0], -1), ord=2, dim=1).mean()
            z_var_after_detokenizer = torch.mean(torch.var(z.view(z.shape[0], -1), dim=0))
            out = {
                "debug/z_norm_before_detokenizer": z_norm_before_detokenizer,
                "debug/z_var_before_detokenizer": z_var_before_detokenizer,
                "debug/z_norm_after_detokenizer": z_norm_after_detokenizer,
                "debug/z_var_after_detokenizer": z_var_after_detokenizer,
            }
            # return
            return z, out
        return z

    def forward_embeddings(self, x: torch.tensor, p: torch.tensor) -> torch.tensor:
        """
        Args:
            x: torch.tensor sequence of weight/channel tokens
            p: torch.tensor sequence of positions
        Returns:
            z: torch.tensor sequence of latent representations
        """
        x = self.forward_encoder(x, p)
        # x = self.model.projection_head(x)
        # x = x.view(x.shape[0], -1)  # flatten
        if self.aggregate_embeddings == "mean":
            x = torch.mean(x, dim=1)  # average
        elif self.aggregate_embeddings == "flatten":
            x = x.view(x.shape[0], -1)  # flatten
        return x


class PositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs.
    Attributes:
        posemb_init: positional embedding initializer.
        max_positions: maximum number of positions to embed.
        embedding_dim: dimension of the input embeddings.
    """

    def __init__(self, max_positions=[48, 256], embedding_dim=128):
        super().__init__()
        self.max_positions = max_positions
        self.embedding_dim = embedding_dim
        if len(max_positions) == 2:
            self.pe1 = nn.Embedding(max_positions[0], embedding_dim // 2)
            self.pe2 = nn.Embedding(max_positions[1], embedding_dim // 2)
            self.pe3 = None
        elif len(max_positions) == 3:
            self.pe1 = nn.Embedding(max_positions[0], embedding_dim // 2)  # add 1 + 2
            self.pe2 = nn.Embedding(max_positions[1], embedding_dim // 2)  # add 1 + 2
            self.pe3 = nn.Embedding(max_positions[2], embedding_dim // 2)  # cat 1+2 & 3

    def forward(self, inputs, pos):
        """Applies the AddPositionEmbs module.
        Args:
            inputs: Inputs to the layer, shape `(batch_size, seq_len, emb_dim)`.
            pos: Position of the first token in each sequence, shape `(batch_size,seq_len,2)`.
        Returns:
            Output tensor with shape `(batch_size, seq_len, emb_dim + 2)`.
        """
        assert (
            inputs.ndim == 3
        ), f"Number of dimensions should be 3, but it is {inputs.ndim}"
        assert pos.shape[2] == len(
            self.max_positions
        ), f"Position tensors should have as many demsions as max_positions. Pos shape: {pos.shape[2]}, MP: {self.max_positions}, LEN: {len(self.max_positions)}"
        assert (
            pos.shape[0] == inputs.shape[0]
        ), "Position tensors should have the same batch size as inputs"
        assert (
            pos.shape[1] == inputs.shape[1]
        ), "Position tensors should have the same seq length as inputs"

        pos_emb1 = self.pe1(pos[:, :, 0])
        pos_emb2 = self.pe2(pos[:, :, 1])
        if self.pe3 is not None:
            pos_emb3 = self.pe3(pos[:, :, 2])
            pos_emb = [pos_emb1 + pos_emb2, pos_emb3]
        else:
            pos_emb = [pos_emb1, pos_emb2]

        pos_emb = torch.cat(pos_emb, dim=2)

        out = inputs + pos_emb
        return out


class ProjectionHead(nn.Module):
    """
    Projection head: maps sequences of token embeddings and maps them to embeddings
    """

    def __init__(
        self, d_model: int = 512, nhead: int = 8, num_layers: int = 6, odim: int = 50
    ):
        super(ProjectionHead, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, odim, bias=False)
        self.comp_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, z: torch.tensor) -> torch.tensor:
        """
        Args:
            z: sequence of token embeddings [nbatch,token_window,token_dim]
        """
        # init compression token
        b, n, _ = z.shape
        copm_tokens = repeat(self.comp_token, "() n d -> b n d", b=b)
        z = torch.cat((copm_tokens, z), dim=1)
        # pass through
        z = self.encoder(z)
        # take only comp_token
        z = z[:, 0, :].squeeze()
        # pass through head
        z = self.head(z)
        # return
        return z


class SimpleProjectionHead(nn.Module):
    """
    Projection head: maps sequences of token embeddings and maps them to embeddings
    """

    def __init__(self, d_model: int = 512, n_tokens: int = 12, odim: int = 50):
        super(SimpleProjectionHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(d_model * n_tokens, odim, bias=False),
            nn.LayerNorm(odim),
            nn.ReLU(),
            nn.Linear(odim, odim, bias=False),
            nn.LayerNorm(odim),
            nn.ReLU(),
        )

    def forward(self, z: torch.tensor) -> torch.tensor:
        """
        Args:
            z: sequence of token embeddings [nbatch,token_window,token_dim]
        """
        # avereage tokens
        # z = z.mean(dim=1)
        z = z.view(z.shape[0], -1)
        # pass through head
        z = self.head(z)
        # return
        return z
    

class FunctionalSinusoidalPositionEmbeddings(nn.Module):
    """Applies sinusoidal positional embeddings with the assumption that the embedding dimension is even."""

    def __init__(self, embedding_dim: int, num_pos_dims: int = 3):
        super(FunctionalSinusoidalPositionEmbeddings, self).__init__()
        assert (
            embedding_dim % 2 == 0
        ), "Embedding dimension must be even to ensure compatibility with sin/cos structure."

        self.embedding_dim = embedding_dim
        self.num_pos_dims = num_pos_dims

    def _get_sinusoidal_embedding(self, positions, embedding_dim):
        """Generates sinusoidal embeddings dynamically for each positional dimension and sums them.

        Args:
            positions: Tensor of positions with shape `(batch_size, seq_len, num_pos_dims)`.
            embedding_dim: The size of the embedding to generate for each position.

        Returns:
            Tensor of shape `(batch_size, seq_len, embedding_dim)` with the summed sinusoidal embeddings.
        """
        batch_size, seq_len, num_pos_dims = (
            positions.size()
        )  # Extract batch size, seq length, and positional dimensions
        emb = torch.zeros(
            batch_size, seq_len, embedding_dim, device=positions.device
        )  # Initialize full embedding

        # Compute sinusoidal embeddings for each positional dimension and sum them up
        for i in range(num_pos_dims):
            position_indices = (
                positions[:, :, i].float().unsqueeze(-1)
            )  # Shape: (batch_size, seq_len, 1)
            div_term = torch.exp(
                torch.arange(0, embedding_dim, 2, device=positions.device).float()
                * (-math.log(10000.0) / embedding_dim)
            )

            # Create embedding for this position dimension
            pos_emb_dim = torch.zeros(
                batch_size, seq_len, embedding_dim, device=positions.device
            )

            # Apply sin to even indices and cos to odd indices
            pos_emb_dim[:, :, 0::2] = torch.sin(
                position_indices * div_term
            )  # Sine on even indices
            pos_emb_dim[:, :, 1::2] = torch.cos(
                position_indices * div_term
            )  # Cosine on odd indices

            # Sum the positional embeddings for each dimension
            emb += pos_emb_dim

        return emb

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Evaluates positional embeddings for inputs (handles both single sample and batch cases).

        Args:
            x: Inputs to the layer, shape `(batch_size, seq_len, emb_dim)` or `(seq_len, emb_dim)`.
            pos: Position tensor of shape `(batch_size, seq_len, num_pos_dims)` or `(seq_len, num_pos_dims)` for single samples.

        Returns:
            Output tensor with shape `(batch_size, seq_len, emb_dim)` or `(seq_len, emb_dim)` for single samples.
        """
        # If the input is not batched, add an extra batch dimension
        single_sample = x.ndim == 2  # Check if the input is a single sample
        if single_sample:
            x = x.unsqueeze(0)  # Add batch dimension
            pos = pos.unsqueeze(0)  # Add batch dimension for position

        # Compute positional embeddings and add them to input
        pos_emb = self._get_sinusoidal_embedding(pos, self.embedding_dim)
        x_with_pos = x + pos_emb

        # If it was a single sample, squeeze the batch dimension back
        if single_sample:
            x_with_pos = x_with_pos.squeeze(0)

        return x_with_pos
