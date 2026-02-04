from utils import *
class MeanVarianceLoss(nn.Module):
    """
    Mean-variance loss between predicted age and target distribution.
    Encourages matching both mean and variance of the age distribution.
    """

    def __init__(self, max_age: int = 100):
        super().__init__()
        ages = torch.arange(0, max_age + 1).float()
        self.register_buffer("ages", ages)

    def forward(
        self, pred_logits: torch.Tensor, target_dist: torch.Tensor
    ) -> torch.Tensor:
        # pred_dist: softmax over age bins
        pred_dist = F.softmax(pred_logits, dim=1)

        # Ensure ages buffer is on the same device as input tensors
        ages = self.ages.to(pred_logits.device)  # (A,)

        # Means
        pred_mean = (pred_dist * ages).sum(dim=1)
        target_mean = (target_dist * ages).sum(dim=1)

        # Variances
        pred_var = (pred_dist * (ages - pred_mean.unsqueeze(1)) ** 2).sum(dim=1)
        target_var = (target_dist * (ages - target_mean.unsqueeze(1)) ** 2).sum(dim=1)

        mean_loss = F.l1_loss(pred_mean, target_mean)
        var_loss = F.l1_loss(pred_var, target_var)

        return mean_loss + var_loss


class AgeLoss(nn.Module):
    """
    Combined loss:
    - L1 (MAE) between predicted age and scalar label
    - MeanVarianceLoss on distribution head
    - KL-divergence between predicted and target age distributions
    """

    def __init__(
        self,
        max_age: int = 100,
        lambda_l1: float = 1.0,
        lambda_meanvar: float = 0.2,
        lambda_kl: float = 0.5,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_meanvar = lambda_meanvar
        self.lambda_kl = lambda_kl

        self.l1 = nn.L1Loss()
        self.meanvar = MeanVarianceLoss(max_age=max_age)

    def forward(
        self,
        age_pred: torch.Tensor,
        dist_logits: Optional[torch.Tensor],
        age_target: torch.Tensor,
        age_dist_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        losses = {}

        l1_loss = self.l1(age_pred, age_target)
        losses["l1"] = l1_loss

        total_loss = self.lambda_l1 * l1_loss

        if dist_logits is not None:
            mv_loss = self.meanvar(dist_logits, age_dist_target)
            losses["meanvar"] = mv_loss
            total_loss = total_loss + self.lambda_meanvar * mv_loss

            # KL divergence: target is "true" distribution
            log_pred = F.log_softmax(dist_logits, dim=1)
            kl = F.kl_div(
                log_pred, age_dist_target, reduction="batchmean", log_target=False
            )
            losses["kl"] = kl
            total_loss = total_loss + self.lambda_kl * kl
        else:
            losses["meanvar"] = torch.tensor(0.0, device=age_pred.device)
            losses["kl"] = torch.tensor(0.0, device=age_pred.device)

        losses["total"] = total_loss
        return total_loss, losses
