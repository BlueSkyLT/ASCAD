"""Training loss: share CE + independence + XOR CE(S) + window regs."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .affine_moments import centered_product3, m3_table


def _ce(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.nll_loss(p.clamp_min(1e-12).log(), y.long())


def window_entropy(w: torch.Tensor) -> torch.Tensor:
    return -(w.clamp_min(1e-12).log() * w).sum(-1).mean()


def soft_byte(p: torch.Tensor) -> torch.Tensor:
    """p: (B, 256) -> expected integer value (B,)."""
    values = torch.arange(256, device=p.device, dtype=p.dtype)
    return (p * values).sum(-1)


HW_TABLE = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.float32)


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / (a.std() * b.std() + 1e-6)


def centered_product(ell_r: torch.Tensor, ell_c: torch.Tensor) -> torch.Tensor:
    """(ℓ_r − mean)(ℓ_C − mean) over the current batch (2nd-order combine)."""
    return (ell_r - ell_r.mean()) * (ell_c - ell_c.mean())


def order_hinge(mu_r: torch.Tensor, mu_c: torch.Tensor, margin: float, trace_len: float) -> torch.Tensor:
    """Penalize μ_r + margin > μ_C. Zero when r is at least `margin` samples left of C."""
    viol = torch.relu(mu_r - mu_c + float(margin))
    scale = max(float(trace_len), 1.0)
    return (viol / scale).square().mean()


def edge_mass_penalty(w: torch.Tensor, edge: int = 12) -> torch.Tensor:
    """Mean attention mass on the first/last `edge` samples (discourages t=0 / t=T-1)."""
    e = max(int(edge), 0)
    if e == 0 or w.shape[-1] <= 2 * e:
        return w.new_zeros(())
    return (w[:, :e].sum(-1) + w[:, -e:].sum(-1)).mean()


def loss_2o(
    ell_r: torch.Tensor,
    ell_c: torch.Tensor,
    s: torch.Tensor,
    corr_mode: str = "pos",
    mu_r: torch.Tensor | None = None,
    mu_c: torch.Tensor | None = None,
    lambda_rep: float = 0.0,
    tau_rep: float = 40.0,
    lambda_order: float = 0.0,
    order_margin: float = 200.0,
    trace_len: float = 700.0,
    w_r: torch.Tensor | None = None,
    w_c: torch.Tensor | None = None,
    lambda_edge: float = 0.0,
    edge: int = 12,
) -> torch.Tensor:
    """HW(S*) correlation of the centered window product. No r/C heads.

    corr_mode:
      pos — maximize +corr (legacy desync0 train)
      neg — maximize −corr (matches oracle 156×517 sign on desync0)
      abs — maximize |corr| (sign-agnostic; preferred when jitter flips sign)

    lambda_order: hinge so μ_r + margin ≤ μ_C (r before C, gap at least margin).
    lambda_edge: keep mass off the trace ends (repulsion-to-edges is the failure mode).
    lambda_rep: old exp(-Δμ²) term; prefer order+edge instead.
    """
    prod = centered_product(ell_r, ell_c)
    hw = HW_TABLE.to(device=s.device, dtype=prod.dtype)[s.long()]
    c = pearson_corr(prod, hw)
    mode = str(corr_mode).lower()
    if mode == "pos":
        loss = -c
    elif mode == "neg":
        loss = c
    elif mode == "abs":
        loss = -c.abs()
    else:
        raise ValueError(f"unknown corr_mode: {corr_mode}")
    if lambda_rep > 0 and mu_r is not None and mu_c is not None:
        d = mu_r - mu_c
        loss = loss + float(lambda_rep) * torch.exp(-d.square() / (2.0 * float(tau_rep) ** 2)).mean()
    if lambda_order > 0 and mu_r is not None and mu_c is not None:
        loss = loss + float(lambda_order) * order_hinge(mu_r, mu_c, order_margin, trace_len)
    if lambda_edge > 0 and w_r is not None and w_c is not None:
        loss = loss + float(lambda_edge) * (edge_mass_penalty(w_r, edge) + edge_mass_penalty(w_c, edge))
    return loss


def loss_3o(
    ell_a: torch.Tensor,
    ell_b: torch.Tensor,
    ell_c: torch.Tensor,
    s: torch.Tensor,
    corr_mode: str = "abs",
    mu_a: torch.Tensor | None = None,
    mu_b: torch.Tensor | None = None,
    mu_c: torch.Tensor | None = None,
    lambda_order: float = 0.0,
    order_margin: float = 200.0,
    trace_len: float = 15000.0,
    w_a: torch.Tensor | None = None,
    w_b: torch.Tensor | None = None,
    w_c: torch.Tensor | None = None,
    lambda_edge: float = 0.0,
    edge: int = 12,
) -> torch.Tensor:
    """Affine 3-way moment correlation objective for ASCADv2.

    Returns a scalar loss that maximizes the correlation between the centered
    product of the three window scores and the affine moment m3(S*). The option
    names match the v1 loss_2o API for a compatible training interface.
    """
    prod = centered_product3(ell_a, ell_b, ell_c)
    target = m3_table().to(device=s.device, dtype=prod.dtype)[s.long()]
    c = pearson_corr(prod, target)
    mode = str(corr_mode).lower()
    if mode == "pos":
        loss = -c
    elif mode == "neg":
        loss = c
    elif mode == "abs":
        loss = -c.abs()
    else:
        raise ValueError(f"unknown corr_mode: {corr_mode}")
    if lambda_order > 0 and mu_a is not None and mu_b is not None and mu_c is not None:
        # Keep alpha/beta before C in the first/second band split.
        loss = loss + float(lambda_order) * (
            order_hinge(mu_a, mu_c, order_margin, trace_len)
            + order_hinge(mu_b, mu_c, order_margin, trace_len)
        )
    if lambda_edge > 0 and w_a is not None and w_b is not None and w_c is not None:
        loss = loss + float(lambda_edge) * (
            edge_mass_penalty(w_a, edge) + edge_mass_penalty(w_b, edge) + edge_mass_penalty(w_c, edge)
        )
    return loss


def loss_head(
    logits: torch.Tensor,
    s: torch.Tensor,
    w: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    lambda_edge: float = 0.0,
    edge: int = 12,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy head loss for the 3-window scalar head.

    logits: (B, 256)
    """
    loss = F.cross_entropy(logits, s.long(), label_smoothing=float(label_smoothing))
    if lambda_edge > 0 and w is not None:
        loss = loss + float(lambda_edge) * edge_mass_penalty(w, edge)
    return loss


def hw_corr_loss(x: torch.Tensor, w: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """−Pearson(sum_t w_t X_t, HW(byte)). First-order localization signal."""
    pooled = (w * x).sum(-1)
    hw = HW_TABLE.to(device=labels.device, dtype=pooled.dtype)[labels.long()]
    p = pooled - pooled.mean()
    h = hw - hw.mean()
    denom = p.std() * h.std() + 1e-6
    return -(p * h).mean() / denom


def independence_loss(p_r: torch.Tensor, p_c: torch.Tensor) -> torch.Tensor:
    """(mean_i (r̃_i - r̄)(C̃_i - C̄))² — batch covariance of predicted bytes."""
    r_tilde = soft_byte(p_r)
    c_tilde = soft_byte(p_c)
    r_c = (r_tilde - r_tilde.mean()) * (c_tilde - c_tilde.mean())
    # Scale so λ_ind=1 is O(1) like CE (byte values are 0..255).
    return (r_c.mean() / 255.0).square()


def total_loss(
    out: dict,
    s: torch.Tensor,
    r: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    lambda_ind: float = 1.0,
    lambda_cons: float = 1.0,
    lambda_ent: float = 1e-3,
    lambda_rep: float = 0.1,
    lambda_hw: float = 1.0,
    tau: float = 40.0,
    x: torch.Tensor | None = None,
):
    parts = {"ce_s": _ce(out["p_s"], s)}
    loss = lambda_cons * parts["ce_s"]
    if r is not None and c is not None and "p_r" in out and "p_c" in out:
        parts["ce_r"] = _ce(out["p_r"], r)
        parts["ce_c"] = _ce(out["p_c"], c)
        loss = loss + parts["ce_r"] + parts["ce_c"]
        if lambda_ind > 0:
            parts["ind"] = independence_loss(out["p_r"], out["p_c"])
            loss = loss + lambda_ind * parts["ind"]
        if lambda_hw > 0 and x is not None and "w_r" in out and "w_c" in out:
            parts["hw"] = hw_corr_loss(x, out["w_r"], r) + hw_corr_loss(x, out["w_c"], c)
            loss = loss + lambda_hw * parts["hw"]
    if lambda_ent > 0 and "w_r" in out and "w_c" in out:
        parts["ent"] = window_entropy(out["w_r"]) + window_entropy(out["w_c"])
        loss = loss + lambda_ent * parts["ent"]
    if lambda_rep > 0 and "mu_r" in out and "mu_c" in out:
        d = out["mu_r"] - out["mu_c"]
        parts["rep"] = torch.exp(-d.square() / (2.0 * tau * tau)).mean()
        loss = loss + lambda_rep * parts["rep"]
    parts["loss"] = loss
    return loss, parts
