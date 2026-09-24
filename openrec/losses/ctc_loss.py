import torch
from torch import nn


def ctc_confidence_weights(
    initial_logits: torch.Tensor,
    confidence_lambda: float = 2.0,
    blank_idx: int = 0,
) -> torch.Tensor:
    """Return detached uncertainty weights for each CTC sample.

    Confidence is the mean maximum posterior over timesteps whose argmax is
    not blank. An all-blank sample falls back to the mean maximum posterior
    over all timesteps, keeping the result finite without using labels.

    Args:
        initial_logits: Frozen initial CTC logits shaped ``[B, T, V]``.
        confidence_lambda: Non-negative uncertainty scaling factor.
        blank_idx: Index of the CTC blank class.

    Returns:
        Detached sample weights shaped ``[B]``.
    """
    if initial_logits.ndim != 3:
        raise ValueError(
            "initial_logits must have shape [B,T,V], got "
            f"{tuple(initial_logits.shape)}"
        )
    if confidence_lambda < 0.0:
        raise ValueError(
            "confidence_lambda must be non-negative, got "
            f"{confidence_lambda}"
        )
    vocab_size = initial_logits.shape[-1]
    if not 0 <= blank_idx < vocab_size:
        raise ValueError(
            f"blank_idx must be in [0, {vocab_size}), got {blank_idx}"
        )

    probabilities = torch.softmax(initial_logits, dim=-1).detach()
    timestep_confidence, predicted_ids = probabilities.max(dim=-1)
    informative_mask = predicted_ids.ne(blank_idx)
    informative_count = informative_mask.sum(dim=-1)
    informative_sum = (
        timestep_confidence * informative_mask.to(timestep_confidence.dtype)
    ).sum(dim=-1)
    informative_mean = informative_sum / informative_count.clamp_min(1)
    all_timestep_mean = timestep_confidence.mean(dim=-1)
    sample_confidence = torch.where(
        informative_count.gt(0),
        informative_mean,
        all_timestep_mean,
    )
    uncertainty = 1.0 - sample_confidence
    return 1.0 + confidence_lambda * uncertainty


class CTCLoss(nn.Module):

    def __init__(
        self,
        use_focal_loss=False,
        zero_infinity=False,
        iterative_ctc_initial_weight=0.3,
        iterative_ctc_confidence_weighting=False,
        iterative_ctc_confidence_lambda=2.0,
        **kwargs,
    ):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0,
                                    reduction='none',
                                    zero_infinity=zero_infinity)
        self.use_focal_loss = use_focal_loss
        if iterative_ctc_initial_weight < 0.0:
            raise ValueError(
                "iterative_ctc_initial_weight must be non-negative, "
                f"got {iterative_ctc_initial_weight}"
            )
        self.iterative_ctc_initial_weight = iterative_ctc_initial_weight
        if iterative_ctc_confidence_lambda < 0.0:
            raise ValueError(
                "iterative_ctc_confidence_lambda must be non-negative, "
                f"got {iterative_ctc_confidence_lambda}"
            )
        self.iterative_ctc_confidence_weighting = bool(
            iterative_ctc_confidence_weighting
        )
        self.iterative_ctc_confidence_lambda = (
            iterative_ctc_confidence_lambda
        )

    def _loss_from_logits(self, predicts, batch):
        """Apply the existing CTC blank, length and focal-loss semantics."""
        batch_size = predicts.size(0)
        label, label_length = batch[1], batch[2]
        predicts = predicts.log_softmax(2)
        predicts = predicts.permute(1, 0, 2)
        preds_lengths = torch.tensor([predicts.size(0)] * batch_size,
                                     dtype=torch.long)
        loss = self.loss_func(predicts, label, preds_lengths, label_length)

        if self.use_focal_loss:
            # Use torch.clamp to limit the range of loss, avoiding overflow in exponential calculation
            clamped_loss = torch.clamp(loss, min=-20, max=20)
            weight = 1 - torch.exp(-clamped_loss)
            weight = torch.square(weight)
            # Use torch.where to avoid multiplying by zero weight
            loss = torch.where(weight > 0, loss * weight, loss)
        return loss.mean()

    def _per_sample_loss_from_logits(self, predicts, batch):
        """Return CTC losses shaped ``[B]`` with existing loss semantics."""
        batch_size = predicts.size(0)
        label, label_length = batch[1], batch[2]
        predicts = predicts.log_softmax(2)
        predicts = predicts.permute(1, 0, 2)
        preds_lengths = torch.tensor(
            [predicts.size(0)] * batch_size,
            dtype=torch.long,
        )
        loss = self.loss_func(
            predicts,
            label,
            preds_lengths,
            label_length,
        )

        if self.use_focal_loss:
            clamped_loss = torch.clamp(loss, min=-20, max=20)
            weight = 1 - torch.exp(-clamped_loss)
            weight = torch.square(weight)
            loss = torch.where(weight > 0, loss * weight, loss)
        return loss

    def forward(self, predicts, batch):
        if isinstance(predicts, dict):
            expected_keys = {"initial_logits", "refined_logits"}
            if not expected_keys.issubset(predicts):
                raise KeyError(
                    "Iterative CTC predictions require initial_logits and "
                    "refined_logits"
                )
            initial_loss = self._loss_from_logits(
                predicts["initial_logits"],
                batch,
            )
            if self.iterative_ctc_confidence_weighting:
                refined_losses = self._per_sample_loss_from_logits(
                    predicts["refined_logits"],
                    batch,
                )
                sample_weights = ctc_confidence_weights(
                    predicts["initial_logits"],
                    confidence_lambda=(
                        self.iterative_ctc_confidence_lambda
                    ),
                    blank_idx=0,
                ).to(
                    device=refined_losses.device,
                    dtype=refined_losses.dtype,
                )
                refined_loss = (
                    sample_weights * refined_losses
                ).sum() / sample_weights.sum()
            else:
                refined_loss = self._loss_from_logits(
                    predicts["refined_logits"],
                    batch,
                )
            loss = (
                refined_loss
                + self.iterative_ctc_initial_weight * initial_loss
            )
            return {
                "loss": loss,
                "loss_initial": initial_loss,
                "loss_refined": refined_loss,
            }

        return {'loss': self._loss_from_logits(predicts, batch)}
