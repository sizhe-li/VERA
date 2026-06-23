# Latent Jacobian: Loss Definitions

This document explains each loss term used to train the latent Jacobian model (encoder, decoder, and Jacobian map J(z)).

**Notation**

- `z_t` = encoder(current frame), `z_next` = encoder(next frame)
- `delta_z_gt` = z_next − z_t (true latent change)
- `J_t` = J(z_t) (Jacobian at current latent)
- `du_gt` = ground-truth action from dataset
- `du_pred` = solution of J_t @ du = delta_z_gt (least-squares)
- `z_dot_pred` = J_t @ du (with du = du_gt or du_pred depending on config)
- `z_pred_next` = z_t + z_dot_pred (forward-predicted next latent)
- Dec(·) = decoder (absolute z → image; relative z_dot → [I_dot, V])
- I_dot = rgb_next − rgb_curr (image difference); V = optical flow (if used)

---

## 1. Backward action (`loss_backward_action`)

**Formula:** MSE(du_pred, du_gt)

**Meaning:** The model solves J_t @ du = delta_z_gt for du. This loss penalizes the predicted action du_pred for differing from the true action du_gt. It trains the Jacobian so that the “inverse” (solve) matches the real action that caused the observed latent change.

**Role:** Ties the Jacobian to the true action; without it, J could be arbitrary as long as forward predictions look good.

---

## 2. Forward absolute (`loss_abs_pred`)

**Formula:** L1(Dec(z_pred_next), Dec(z_next))

**Meaning:** The forward-predicted next latent z_pred_next = z_t + J_t @ du should decode to the same image as the true next latent z_next. So “Dec(z_t + J_t @ du)” should match “Dec(z_next)”.

**Role:** Encourages the linearized dynamics z_next ≈ z_t + J_t @ du in **decoded space**: the predicted next frame (from J and du) should match the decoded true next frame.

---

## 3. Forward relative (`loss_rel_pred`)

**Formula:** L1(Dec(z_dot_pred), Dec(z_dot_gt))

**Meaning:** The latent delta predicted by the Jacobian, z_dot_pred = J_t @ du, is decoded and compared to the decoded true latent delta z_dot_gt = delta_z_gt. So “Dec(J_t @ du)” should match “Dec(delta_z_gt)”.

**Role:** This is the main **flow-like** penalty on the Jacobian: in decoded space, the model’s motion (from J @ du) must match the true motion (from delta_z). Analogous to supervising optical flow in image-space Jacobian methods. Often weighted heavily (e.g. 2000) so J is not dominated by reconstruction.

---

## 4. z_dot decode (`loss_z_dot_decode`)

**Formula:** L1(Dec(z_dot_gt), [I_dot, V])

**Meaning:** The decoder is trained so that when given the **true** latent delta z_dot_gt, it outputs the image difference and flow: Dec(z_dot_gt) should match [I_dot, V] (I_dot = rgb_next − rgb_curr, V = flow if available).

**Role:** Teaches the decoder the correct interpretation of “relative” latent input: latent deltas should map to image motion and flow. This gives a direct **flow signal** in latent space (like flow loss in image Jacobian). Also stabilizes Dec so that rel_pred and abs_pred are meaningful.

---

## 5. Reconstruction (`loss_recon`)

**Formula:** L1(Dec(z_t)[:3], rgb_curr)

**Meaning:** The current latent z_t should decode to the current image. Only the first 3 (RGB) channels are used; flow channels, if any, are not supervised here.

**Role:** Standard autoencoding of the current frame. Ensures the encoder and decoder learn a good latent representation before the Jacobian/flow losses can be effective.

---

## Summary table

| Loss               | What is compared | Purpose |
|--------------------|------------------|--------|
| backward_action    | du_pred vs du_gt | Jacobian inverse matches true action |
| abs_pred           | Dec(z_pred_next) vs Dec(z_next) | Forward prediction in image space |
| rel_pred           | Dec(J@du) vs Dec(delta_z_gt) | Jacobian in flow/decoded space (key for J) |
| z_dot_decode       | Dec(delta_z_gt) vs [I_dot, V] | Decoder maps latent deltas to flow |
| recon              | Dec(z_t) vs rgb_curr | Autoencode current frame |

---

## Weighting

Default weights are set so that Jacobian-related losses (backward_action, abs_pred, rel_pred, z_dot_decode) are not dwarfed by recon. In particular, `weight_rel_pred` and `weight_z_dot_decode` are kept large so the Jacobian is penalized via flow-like signals even while the encoder/decoder learn first.
