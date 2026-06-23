import torch


def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    h = h.clamp(0.0, 1.0)
    s = s.clamp(0.0, 1.0)
    v = v.clamp(0.0, 1.0)

    hp = h * 6.0
    i = torch.floor(hp).to(torch.int64)
    f = hp - i.to(hp.dtype)
    i = i % 6

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = torch.where(
        i == 0,
        v,
        torch.where(
            i == 1,
            q,
            torch.where(
                i == 2,
                p,
                torch.where(i == 3, p, torch.where(i == 4, t, v)),
            ),
        ),
    )
    g = torch.where(
        i == 0,
        t,
        torch.where(
            i == 1,
            v,
            torch.where(
                i == 2,
                v,
                torch.where(i == 3, q, torch.where(i == 4, p, p)),
            ),
        ),
    )
    b = torch.where(
        i == 0,
        p,
        torch.where(
            i == 1,
            p,
            torch.where(
                i == 2,
                t,
                torch.where(i == 3, v, torch.where(i == 4, v, q)),
            ),
        ),
    )

    return torch.stack([r, g, b], dim=-1)


def flow_to_rgb(
    flow: torch.Tensor, max_magnitude: torch.Tensor | float | None = None
) -> torch.Tensor:
    if not isinstance(flow, torch.Tensor) or flow.dim() not in (4, 5):
        raise ValueError(
            "Expected flow tensor with shape [T,2,H,W], [B,2,T,H,W], or [B,T,2,H,W], "
            f"got {type(flow)} with dim={getattr(flow, 'dim', lambda: None)()}."
        )

    if flow.dim() == 4:
        if flow.shape[1] != 2:
            raise ValueError(f"Expected 2-channel flow in [T,2,H,W], got shape {tuple(flow.shape)}.")
        u = flow[:, 0]
        v = flow[:, 1]
    else:
        if flow.shape[1] == 2:
            u = flow[:, 0]
            v = flow[:, 1]
        elif flow.shape[2] == 2:
            u = flow[:, :, 0]
            v = flow[:, :, 1]
        else:
            raise ValueError(
                f"Expected 2-channel flow in [B,2,T,H,W] or [B,T,2,H,W], got shape {tuple(flow.shape)}."
            )

    angle = torch.atan2(v, u)
    h = (angle + torch.pi) / (2.0 * torch.pi)
    mag = torch.sqrt(u.square() + v.square())

    if max_magnitude is None:
        if flow.dim() == 4:
            max_magnitude = mag.amax(dim=(0, 1, 2), keepdim=True)
        else:
            max_magnitude = mag.amax(dim=(1, 2, 3), keepdim=True)
    if isinstance(max_magnitude, (float, int)):
        max_magnitude = torch.tensor(float(max_magnitude), device=flow.device, dtype=flow.dtype)
    if isinstance(max_magnitude, torch.Tensor):
        max_magnitude = max_magnitude.to(device=flow.device, dtype=flow.dtype)

    max_magnitude = max_magnitude.clamp_min(1e-6)
    val = (mag / max_magnitude).clamp(0.0, 1.0)
    sat = torch.ones_like(val)

    rgb = hsv_to_rgb(h, sat, val)
    if flow.dim() == 4:
        return rgb.permute(0, 3, 1, 2).contiguous()
    return rgb.permute(0, 4, 1, 2, 3).contiguous()
