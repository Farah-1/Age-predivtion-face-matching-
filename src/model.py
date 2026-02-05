from utils import * 

class AgePredictionModel(nn.Module):
    """
    EfficientNet-B4 backbone with multi-task head:
    - Age regression (single scalar)
    - Optional age distribution (soft-label) branch
    """

    def __init__(
        self,
        use_soft_labels: bool = True,
        max_age: int = 100,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.use_soft_labels = use_soft_labels
        self.max_age = max_age

        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        in_features = self.backbone.num_features
        self.dropout = nn.Dropout(p=dropout)

        self.age_regressor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        if self.use_soft_labels:
            self.age_distribution_head = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, max_age + 1),
            )
        else:
            self.age_distribution_head = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feats = self.backbone(x)
        feats = self.dropout(feats)
        age_pred = self.age_regressor(feats).squeeze(-1)
        if self.use_soft_labels and self.age_distribution_head is not None:
            logits = self.age_distribution_head(feats)
        else:
            logits = None
        return age_pred, logits
