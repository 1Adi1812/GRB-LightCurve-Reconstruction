import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mambular.arch_utils.mamba_utils.mamba_arch import MambaBlock
from mambular.base_models.utils.basemodel import BaseModel
from mambular.base_models.mambular import Mambular
from mambular.utils.get_feature_dimensions import get_feature_dimensions
from torch.utils.data import DataLoader, Dataset


class MambaRegressor(nn.Module):
    def __init__(
        self,
        output_dim=1,
        dt_rank=32,
        d_model=1,
        expand_factor=8,
        d_state=32,
        dropout=0.01,
        bidirectional=True,
        activation=nn.SiLU
    ):
        super(MambaRegressor, self).__init__()

        self.d_model = d_model
        self.expand = nn.Linear(1, d_model)

        self.mamba_branch = MambaBlock(
            dt_rank=dt_rank,
            d_model=d_model,
            expand_factor=expand_factor,
            d_state=d_state,
            dropout=dropout,
            bidirectional=bidirectional,
            activation=activation
        )

        total_features = (
            self.mamba_branch.out_proj.out_features
        )
        self.act = activation
        self.fc = nn.Linear(total_features, output_dim)

    def forward(self, x):
        b1 = self.expand(x)
        b1 = self.mamba_branch(b1)

        output = self.fc(b1)
        return output


if __name__ == "__main__":
    model = MambaRegressor(
        output_dim=1, d_model=2, expand_factor=16, bidirectional=True
    ).to("cuda")

    t = torch.randn([2, 17, 1]).to("cuda")
    o = model(t)
    print(t.detach().cpu().shape)
    print(o.detach().cpu().shape)
