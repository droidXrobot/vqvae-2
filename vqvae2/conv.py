import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log2

from .utils import HelperModule
from typing import Callable, Dict, Optional, List, Iterable
from functools import partial


class Residual(HelperModule):
    def build(self, net: nn.Module, use_rezero: bool = False, rezero_init: float = 0.0):
        self.net = net
        self.alpha = nn.Parameter(torch.FloatTensor([rezero_init])) if use_rezero else 1.0

    def forward(self, x):
        return self.net(x) * self.alpha + x


class ResidualLayer(HelperModule):
    def build(
        self,
        in_dim: int,
        residual_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,  # TODO: padding='same'
        bias: bool = True,
        use_batch_norm: bool = True,
        use_rezero: bool = False,
        activation: nn.Module = nn.SiLU,
    ):
        self.activation = activation()
        layers = nn.Sequential(
            nn.Conv2d(
                in_dim,
                residual_dim,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(residual_dim) if use_batch_norm else nn.Identity(),
            activation(),
            nn.Conv2d(residual_dim, in_dim, 1, stride=1, bias=bias),
            nn.BatchNorm2d(in_dim) if use_batch_norm else nn.Identity(),
        )
        self.layers = Residual(layers, use_rezero=use_rezero, rezero_init=0.0)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.activation(self.layers(x))


class ConvDown(HelperModule):
    def build(
        self,
        in_dim: int,
        hidden_dim: int,
        residual_dim: int,
        resample_factor: int,
        num_residual_layers: int = 2,
        resample_method: str = "conv",  # 'max', 'conv', 'auto'
        residual_kernel_size: int = 3,
        residual_stride: int = 1,
        residual_padding: int = 1,
        residual_bias: bool = True,
        use_batch_norm: bool = True,
        use_rezero: bool = False,
        activation: nn.Module = nn.SiLU,
    ):
        assert log2(resample_factor).is_integer(), f"Downsample factor must be a power of 2! Got '{resample_factor}'"

        if resample_method == "auto":
            resample_method = "max" if resample_factor == 2 else "conv"

        if resample_method == "conv":
            self.downsample = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Conv2d(
                            in_dim if i == 0 else hidden_dim,
                            hidden_dim,
                            4,
                            stride=2,
                            padding=1,
                        ),
                        nn.BatchNorm2d(hidden_dim) if use_batch_norm else nn.Identity(),
                        activation(),
                    )
                    for i in range(int(log2(resample_factor)))
                ]
            )
        elif resample_method == "max":
            self.downsample = nn.MaxPool2d(resample_factor, stride=resample_factor)
        else:
            raise ValueError(f"Unknown resample method '{resample_method}'!")

        self.residual = nn.Sequential(
            *[
                ResidualLayer(
                    in_dim=hidden_dim,
                    residual_dim=residual_dim,
                    kernel_size=residual_kernel_size,
                    stride=residual_stride,
                    padding=residual_padding,
                    bias=residual_bias,
                    use_batch_norm=use_batch_norm,
                    use_rezero=use_rezero,
                    activation=activation,
                )
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x, c: Optional[torch.FloatTensor] = None):
        x = self.downsample(x)
        c = c if c is not None else 0.0

        # TODO: for now just add, later we can try concat method
        return self.residual(x + c)


class ConvUp(HelperModule):
    def build(
        self,
        in_dim: int,
        hidden_dim: int,
        residual_dim: int,
        resample_factor: int,
        resample_method: str = "conv",  # 'max', 'conv', 'auto'
        num_residual_layers: int = 2,
        residual_kernel_size: int = 3,
        residual_stride: int = 1,
        residual_padding: int = 1,
        residual_bias: bool = True,
        use_batch_norm: bool = True,
        use_rezero: bool = False,
        conditioning_resample_factors: Optional[List[int]] = None,
        post_concat_kernel_size: int = 5,
        activation: nn.Module = nn.SiLU,
    ):
        assert log2(resample_factor).is_integer(), f"Downsample factor must be a power of 2! Got '{resample_factor}'"

        self.conv_in = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 1, stride=1), activation())

        self.residual = nn.Sequential(
            *[
                ResidualLayer(
                    in_dim=hidden_dim,
                    residual_dim=residual_dim,
                    kernel_size=residual_kernel_size,
                    stride=residual_stride,
                    padding=residual_padding,
                    bias=residual_bias,
                    use_batch_norm=use_batch_norm,
                    use_rezero=use_rezero,
                    activation=activation,
                )
                for _ in range(num_residual_layers)
            ]
        )

        conditioning_resample_factors = conditioning_resample_factors if conditioning_resample_factors else []
        if resample_method == "auto":
            resample_method = "conv"  # interpolate probably isn't good, so just make 'auto' be conv on upsample

        if resample_method == "conv":
            self.conditioning_upsamplers = nn.ModuleList()
            self.upsample = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
                        nn.BatchNorm2d(hidden_dim) if use_batch_norm else nn.Identity(),
                        activation(),
                    )
                    for _ in range(int(log2(resample_factor)))
                ]
            )
            for f in conditioning_resample_factors:
                self.conditioning_upsamplers.append(
                    nn.Sequential(
                        *[
                            nn.Sequential(
                                nn.ConvTranspose2d(hidden_dim if i else in_dim, hidden_dim, 4, stride=2, padding=1),
                                nn.BatchNorm2d(hidden_dim) if use_batch_norm else nn.Identity(),
                                activation(),
                            )
                            for i in range(int(log2(f)))
                        ]
                    )
                )

        # elif resample_method == "interpolate":
        #     self.conditioning_upsamplers = [] if len(conditioning_resample_factors) else None
        #     self.upsample = partial(
        #         F.interpolate,
        #         scale_factor=(resample_factor, resample_factor),
        #         mode="bilinear",
        #     )
        #     for f in conditioning_resample_factors:
        #         self.conditioning_upsamplers.append(
        #             partial(
        #                 F.interpolate,
        #                 scale_factor=(resample_factor, resample_factor),
        #                 mode="bilinear",
        #             )
        #         )
        else:
            raise ValueError(f"Unknown resample method '{resample_method}'!")

        self.conditioning_post_concat = nn.Identity()
        if self.conditioning_upsamplers:
            self.conditioning_post_concat = nn.Sequential(
                nn.Conv2d(
                    (1 + len(conditioning_resample_factors)) * hidden_dim,
                    hidden_dim,
                    post_concat_kernel_size,
                    stride=1,
                    padding="same",
                ),
                nn.BatchNorm2d(hidden_dim) if use_batch_norm else nn.Identity(),
                activation(),
            )

    def forward(self, x, cs: Optional[Iterable[torch.FloatTensor]] = None):
        assert bool(cs) == bool(self.conditioning_upsamplers)
        assert len(cs) == len(self.conditioning_upsamplers)
        cs = cs if cs else []
        x = self.conv_in(x)
        if cs:
            cs = [f(c) for f, c in zip(self.conditioning_upsamplers, cs)]
        x = torch.cat([x, *cs], dim=1)
        x = self.conditioning_post_concat(x)

        x = self.residual(x)
        return self.upsample(x)


if __name__ == "__main__":
    down = ConvDown(8, 16, 32, 2)
    up = ConvUp(8, 16, 32, 2)

    x = torch.randn(4, 8, 32, 32)
    z = down(x)
    y = up(z[:, :8])

    print(x.shape)
    print(z.shape)
    print(y.shape)
