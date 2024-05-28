"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from typing import Tuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from einops import rearrange

from idisc.utils import (AttentionLayer, AttentionLayer2Vis, PositionEmbeddingSine,
                         _get_activation_cls, get_norm)


class ISDHead(nn.Module):
    def __init__(
        self,
        depth: int,
        pixel_dim: int = 256,
        query_dim: int = 256,
        num_heads: int = 4,
        output_dim: int = 1,
        expansion: int = 2,
        activation: str = "silu",
        norm: str = "LN",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.depth = depth
        self.eps = eps
        self.pixel_pe = PositionEmbeddingSine(pixel_dim // 2, normalize=True)
        for i in range(self.depth):
            setattr(
                self,
                f"cross_attn_{i+1}",
                AttentionLayer(
                    sink_dim=pixel_dim,
                    hidden_dim=pixel_dim,
                    source_dim=query_dim,
                    output_dim=pixel_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                    pre_norm=True,
                    sink_competition=False,
                ),
            )
            setattr(
                self,
                f"mlp_{i+1}",
                nn.Sequential(
                    get_norm(norm, pixel_dim),
                    nn.Linear(pixel_dim, expansion * pixel_dim),
                    _get_activation_cls(activation),
                    nn.Linear(expansion * pixel_dim, pixel_dim),
                ),
            )
        setattr(
            self,
            "proj_output",
            nn.Sequential(
                get_norm(norm, pixel_dim),
                nn.Linear(pixel_dim, pixel_dim),
                get_norm(norm, pixel_dim),
                nn.Linear(pixel_dim, output_dim),
            ),
        )

    def forward(self, feature_map: torch.Tensor, idrs: torch.Tensor):
        b, c, h, w = feature_map.shape
        feature_map = rearrange(
            feature_map + self.pixel_pe(feature_map), "b c h w -> b (h w) c"
        )

        for i in range(self.depth):
            update = getattr(self, f"cross_attn_{i+1}")(feature_map.clone(), idrs)
            feature_map = feature_map + update
            feature_map = feature_map + getattr(self, f"mlp_{i+1}")(feature_map.clone())
        out = getattr(self, "proj_output")(feature_map)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)

        feature_map = rearrange(feature_map, "b (h w) c -> b c h w", h=h, w=w)
        return out, feature_map


class ISD(nn.Module):
    def __init__(
        self,
        num_resolutions,
        depth,
        pixel_dim=128,
        query_dim=128,
        num_heads: int = 4,
        output_dim: int = 1,
        expansion: int = 2,
        activation: str = "silu",
        norm: str = "torchLN",
    ):
        super().__init__()
        self.num_resolutions = num_resolutions
        for i in range(num_resolutions):
            setattr(
                self,
                f"head_{i+1}",
                ISDHead(
                    depth=depth,
                    pixel_dim=pixel_dim,
                    query_dim=query_dim,
                    num_heads=num_heads,
                    output_dim=output_dim,
                    expansion=expansion,
                    activation=activation,
                    norm=norm,
                ),
            )

    def forward(
        self, xs: Tuple[torch.Tensor, ...], idrs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        axs[0].set_xlabel("Selected Points")
        axs[0].set_ylabel("Depth")
        axs[1].set_xlabel("Selected Points")
        axs[1].set_ylabel("Logit Norm")
        outs, attns = [], []
        for i in range(self.num_resolutions):
            out, feat = getattr(self, f"head_{i+1}")(xs[i], idrs[i])
            outs.append(out)
            
            # plot the depth values at certain positions
            b, c, h, w = out.shape
            x1 = int(w*0.25)
            x2 = int(w*0.75)
            y1 = int(h*0.6)
            y2 = int(h*0.8)
            out_norm = feat.detach().cpu().norm(dim=1, keepdim=True).numpy()
            n_vec = [out_norm[0, 0, y1, x1], out_norm[0, 0, y1, x2], out_norm[0, 0, y2, x1], out_norm[0, 0, y2, x2]]
            out2vis = out.detach().cpu().numpy()
            d_vec = [out2vis[0, 0, y1, x1], out2vis[0, 0, y1, x2], out2vis[0, 0, y2, x1], out2vis[0, 0, y2, x2]]
            axs[0].plot(d_vec)
            axs[1].plot(n_vec)
        
        axs[0].legend([f"Res {i+1}" for i in range(self.num_resolutions)])
        axs[1].legend([f"Res {i+1}" for i in range(self.num_resolutions)])
        plt.show()

            
        return tuple(outs)

    @classmethod
    def build(cls, config):
        obj = cls(
            num_resolutions=config["model"]["isd"]["num_resolutions"],
            depth=config["model"]["isd"]["depths"],
            pixel_dim=config["model"]["pixel_decoder"]["hidden_dim"],
            query_dim=config["model"]["afp"]["latent_dim"],
            output_dim=config["model"]["output_dim"],
            num_heads=config["model"]["num_heads"],
            expansion=config["model"]["expansion"],
            activation=config["model"]["activation"],
        )
        return obj


class AFP(nn.Module):
    def __init__(
        self,
        num_resolutions: int,
        depth: int = 3,
        pixel_dim: int = 256,
        latent_dim: int = 256,
        num_latents: int = 128,
        num_heads: int = 4,
        activation: str = "silu",
        norm: str = "torchLN",
        expansion: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_resolutions = num_resolutions
        self.iters = depth
        self.num_slots = num_latents
        self.latent_dim = latent_dim
        self.pixel_dim = pixel_dim
        self.eps = eps

        bottlenck_dim = expansion * latent_dim
        for i in range(self.num_resolutions):
            setattr(
                self,
                f"pixel_pe_{i+1}",
                PositionEmbeddingSine(pixel_dim // 2, normalize=True),
            )
            setattr(
                self,
                f"mu_{i+1}",
                nn.Parameter(torch.randn(1, self.num_slots, latent_dim)),
            )

        # Set up attention iterations
        for j in range(self.iters):
            for i in range(self.num_resolutions):
                setattr(
                    self,
                    f"cross_attn_{i+1}_d{1}",
                    AttentionLayer2Vis(
                        sink_dim=latent_dim,
                        hidden_dim=latent_dim,
                        source_dim=pixel_dim,
                        output_dim=latent_dim,
                        num_heads=num_heads,
                        dropout=0.0,
                        pre_norm=True,
                        sink_competition=True,
                    ),
                )
                setattr(
                    self,
                    f"mlp_cross_{i+1}_d{1}",
                    nn.Sequential(
                        get_norm(norm, latent_dim),
                        nn.Linear(latent_dim, bottlenck_dim),
                        _get_activation_cls(activation),
                        nn.Linear(bottlenck_dim, latent_dim),
                    ),
                )

    def forward(
        self, feature_maps: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        b, *_ = feature_maps[0].shape
        idrs = []
        attns = []
        feature_maps_flat = []
        for i in range(self.num_resolutions):
            # feature maps embedding pre-process
            feature_map, (h, w) = feature_maps[i], feature_maps[i].shape[-2:]
            feature_maps_flat.append(
                rearrange(
                    feature_map + getattr(self, f"pixel_pe_{i+1}")(feature_map),
                    "b d h w-> b (h w) d",
                )
            )
            # IDRs generation
            idrs.append(getattr(self, f"mu_{i+1}").expand(b, -1, -1))

        # layers
        for i in range(self.num_resolutions):
            for iter in range(self.iters):
                # Cross attention ops
                cur_idr, cur_attn = getattr(self, f"cross_attn_{i+1}_d{1}")(
                    idrs[i].clone(), feature_maps_flat[i]
                )
                
                idrs[i] = idrs[i] + cur_idr
                idrs[i] = idrs[i] + getattr(self, f"mlp_cross_{i+1}_d{1}")(
                    idrs[i].clone()
                )
                
                # Visualize cur_attn as heat maps
                (h, w) = feature_maps[i].shape[-2:]
                cur_attn = rearrange(cur_attn, "b n (h w) -> b n h w", h=h, w=w)
                cur_attn = cur_attn[:, 0::8, :, :]  # Select elements from dim 1 with interval 8
                cur_attn = cur_attn.detach().cpu().numpy()  # Convert to numpy array

                fig, axs = plt.subplots(2, 2)  # Create a 2x2 grid of subplots
                for ii, ax in enumerate(axs.flat):
                    ax.imshow(cur_attn[0, ii, :, :], cmap='hot')  # Display the i-th attention map
                    ax.axis('off')  # Turn off axis labels
                    axs[ii // 2, ii % 2].set_title(f'Token {ii*8}')
                plt.suptitle(f"Resolution {i}, Iteration {iter}")
                plt.tight_layout()  # Add tight layout to the plot
                plt.colorbar(axs[0, 0].imshow(cur_attn[0, 0, :, :], cmap='hot'), ax=axs, orientation='horizontal')  # Add a colorbar to the plot
                plt.show()  # Show the plot
                
                # Plotting the normal idrs[i, 0, d] over d
                plt.figure()
                idr = idrs[i]
                plt.plot(idr[0].norm(dim=-1).detach().cpu().numpy())
                plt.xlabel("token index")
                plt.ylabel("L2 norm")
                plt.title(f"L2 Norm of IDRs, Res {i}, Iter {iter}")
                plt.show()

        return tuple(idrs)

    @classmethod
    def build(cls, config):
        output_num_resolutions = (
            len(config["model"]["pixel_encoder"]["embed_dims"])
            - config["model"]["afp"]["context_low_resolutions_skip"]
        )
        obj = cls(
            num_resolutions=output_num_resolutions,
            depth=config["model"]["afp"]["depths"],
            pixel_dim=config["model"]["pixel_decoder"]["hidden_dim"],
            num_latents=config["model"]["afp"]["num_latents"],
            latent_dim=config["model"]["afp"]["latent_dim"],
            num_heads=config["model"]["num_heads"],
            expansion=config["model"]["expansion"],
            activation=config["model"]["activation"],
        )
        return obj
