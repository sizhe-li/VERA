import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb as hsv2rgb
from sklearn.decomposition import PCA

EPS = 1e-6


def sub2ind(height, width, y, x):
    return y * width + x


def ind2sub(height, width, ind):
    y = ind // width
    x = ind % width
    return y, x


def get_lr_str(lr):
    lrn = "%.1e" % lr  # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1]  # e.g., 5e-4
    return lrn


def strnum(x):
    s = "%g" % x
    if "." in s:
        if x < 1.0:
            s = s[s.index(".") :]
        s = s[: min(len(s), 4)]
    return s


def assert_same_shape(t1, t2):
    for x, y in zip(list(t1.shape), list(t2.shape)):
        assert x == y


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_stats(name, tensor):
    shape = tensor.shape
    tensor = tensor.detach().cpu().numpy()
    print(
        "%s (%s) min = %.2f, mean = %.2f, max = %.2f"
        % (name, tensor.dtype, np.min(tensor), np.mean(tensor), np.max(tensor)),
        shape,
    )


def normalize_single(d):
    # d is a whatever shape torch tensor
    dmin = torch.min(d)
    dmax = torch.max(d)
    d = (d - dmin) / (EPS + (dmax - dmin))
    return d


def normalize(d):
    # d is B x whatever. normalize within each element of the batch
    out = torch.zeros(d.size(), dtype=d.dtype, device=d.device)
    B = list(d.size())[0]
    for b in list(range(B)):
        out[b] = normalize_single(d[b])
    return out


def meshgrid2d(B, Y, X, stack=False, norm=False, device="cuda", on_chans=False):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y - 1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2d(grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        if on_chans:
            grid = torch.stack([grid_x, grid_y], dim=1)
        else:
            grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x


def gridcloud2d(B, Y, X, norm=False, device: str | torch.device = "cuda"):
    # we want to sample for each location in the grid
    grid_y, grid_x = meshgrid2d(B, Y, X, norm=norm, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    # these are B x N
    xy = torch.stack([x, y], dim=2)
    # this is B x N x 2
    return xy


def reduce_masked_mean(x, mask, dim=None, keepdim=False, broadcast=False):
    # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
    # returns shape-1
    # axis can be a list of axes
    if not broadcast:
        for a, b in zip(x.size(), mask.size()):
            if not a == b:
                print("some shape mismatch:", x.shape, mask.shape)
            assert a == b  # some shape mismatch!
    # assert(x.size() == mask.size())
    prod = x * mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS + torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS + torch.sum(mask, dim=dim, keepdim=keepdim)
    mean = numer / denom
    return mean


def reduce_masked_median(x, mask, keep_batch=False):
    # x and mask are the same shape
    assert x.size() == mask.size()
    device = x.device

    B = list(x.shape)[0]
    x = x.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    if keep_batch:
        x = np.reshape(x, [B, -1])
        mask = np.reshape(mask, [B, -1])
        meds = np.zeros([B], np.float32)
        for b in list(range(B)):
            xb = x[b]
            mb = mask[b]
            if np.sum(mb) > 0:
                xb = xb[mb > 0]
                meds[b] = np.median(xb)
            else:
                meds[b] = np.nan
        meds = torch.from_numpy(meds).to(device)
        return meds.float()
    else:
        x = np.reshape(x, [-1])
        mask = np.reshape(mask, [-1])
        if np.sum(mask) > 0:
            x = x[mask > 0]
            med = np.median(x)
        else:
            med = np.nan
        med = np.array([med], np.float32)
        med = torch.from_numpy(med).to(device)
        return med.float()


def _convert(input_, type_):
    return {
        "float": input_.float(),
        "double": input_.double(),
    }.get(type_, input_)


def _generic_transform_sk_3d(transform, in_type="", out_type=""):
    def apply_transform_individual(input_):
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)

        input_ = input_.permute(1, 2, 0).detach().numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(2, 0, 1)
        output = _convert(output, out_type)
        return output.to(device)

    def apply_transform(input_):
        to_stack = []
        for image in input_:
            to_stack.append(apply_transform_individual(image))
        return torch.stack(to_stack)

    return apply_transform


hsv_to_rgb = _generic_transform_sk_3d(hsv2rgb)


def flow2color(flow, clip=0.0):
    B, C, H, W = list(flow.size())
    assert C == 2
    flow = flow[0:1].detach()
    if clip == 0:
        clip = torch.max(torch.abs(flow)).item()
    flow = torch.clamp(flow, -clip, clip) / clip
    radius = torch.sqrt(torch.sum(flow**2, dim=1, keepdim=True))  # B,1,H,W
    radius_clipped = torch.clamp(radius, 0.0, 1.0)
    angle = torch.atan2(-flow[:, 1:2], -flow[:, 0:1]) / np.pi  # B,1,H,W
    hue = torch.clamp((angle + 1.0) / 2.0, 0.0, 1.0)
    saturation = torch.ones_like(hue) * 0.75
    value = radius_clipped
    hsv = torch.cat([hue, saturation, value], dim=1)  # B,3,H,W
    flow = hsv_to_rgb(hsv)
    flow = (flow * 255.0).type(torch.ByteTensor)
    return flow


FILE_LOC = Path(__file__)
COLORMAP_FILE = str(FILE_LOC.parent / "bremm.png")

# COLORMAP_FILE = str(FILE_LOC / "./utils/bremm.png")


class ColorMap2d:
    def __init__(self, filename=None):
        self._colormap_file = filename or COLORMAP_FILE
        self._img = (plt.imread(self._colormap_file) * 255).astype(np.uint8)

        self._height = self._img.shape[0]
        self._width = self._img.shape[1]

    def __call__(self, X):
        assert len(X.shape) == 2
        output = np.zeros((X.shape[0], 3), dtype=np.uint8)
        for i in range(X.shape[0]):
            x, y = X[i, :]
            xp = int((self._width - 1) * x)
            yp = int((self._height - 1) * y)
            xp = np.clip(xp, 0, self._width - 1)
            yp = np.clip(yp, 0, self._height - 1)
            output[i, :] = self._img[yp, xp]
        return output


def get_2d_colors(xys, H, W):
    N, D = xys.shape
    assert D == 2
    bremm = ColorMap2d()
    xys[:, 0] /= float(W - 1)
    xys[:, 1] /= float(H - 1)
    colors = bremm(xys)
    # print('colors', colors)
    # colors = (colors[0]*255).astype(np.uint8)
    # colors = (int(colors[0]),int(colors[1]),int(colors[2]))
    return colors


def get_n_colors(N, sequential=False):
    label_colors = []
    for ii in range(N):
        if sequential:
            rgb = cm.winter(ii / (N - 1))
            rgb = (np.array(rgb) * 255).astype(np.uint8)[:3]
        else:
            rgb = np.zeros(3)
            while np.sum(rgb) < 128:  # ensure min brightness
                rgb = np.random.randint(0, 256, 3)
        label_colors.append(rgb)
    return label_colors


def pca_embed(emb, keep, valid=None):
    # helper function for reduce_emb
    # emb is B,C,H,W
    # keep is the number of principal components to keep
    emb = emb + EPS
    emb = emb.permute(0, 2, 3, 1).cpu().detach().numpy()  # this is B x H x W x C

    if valid:
        valid = valid.cpu().detach().numpy().reshape((H * W))

    emb_reduced = list()

    B, H, W, C = np.shape(emb)
    for img in emb:
        if np.isnan(img).any():
            emb_reduced.append(np.zeros([H, W, keep]))
            continue

        pixels_kd = np.reshape(img, (H * W, C))

        if valid:
            pixels_kd_pca = pixels_kd[valid]
        else:
            pixels_kd_pca = pixels_kd

        P = PCA(keep)
        P.fit(pixels_kd_pca)

        if valid:
            pixels3d = P.transform(pixels_kd) * valid
        else:
            pixels3d = P.transform(pixels_kd)

        out_img = np.reshape(pixels3d, [H, W, keep]).astype(np.float32)
        if np.isnan(out_img).any():
            emb_reduced.append(np.zeros([H, W, keep]))
            continue

        emb_reduced.append(out_img)

    emb_reduced = np.stack(emb_reduced, axis=0).astype(np.float32)

    return torch.from_numpy(emb_reduced).permute(0, 3, 1, 2)


def pca_embed_together(emb, keep):
    # emb is B,C,H,W
    # keep is the number of principal components to keep
    emb = emb + EPS
    emb = (
        emb.permute(0, 2, 3, 1).cpu().detach().float().numpy()
    )  # this is B x H x W x C

    B, H, W, C = np.shape(emb)
    if np.isnan(emb).any():
        return torch.zeros(B, keep, H, W)

    pixelskd = np.reshape(emb, (B * H * W, C))
    P = PCA(keep)
    P.fit(pixelskd)
    pixels3d = P.transform(pixelskd)
    out_img = np.reshape(pixels3d, [B, H, W, keep]).astype(np.float32)

    if np.isnan(out_img).any():
        return torch.zeros(B, keep, H, W)

    return torch.from_numpy(out_img).permute(0, 3, 1, 2)


def reduce_emb(emb, valid=None, inbound=None, together=False):
    S, C, H, W = list(emb.size())
    keep = 4

    if together:
        reduced_emb = pca_embed_together(emb, keep)
    else:
        reduced_emb = pca_embed(emb, keep, valid)  # not im

    reduced_emb = reduced_emb[:, 1:]
    reduced_emb = normalize(reduced_emb) - 0.5
    if inbound is not None:
        emb_inbound = emb * inbound
    else:
        emb_inbound = None

    return reduced_emb, emb_inbound


def get_feat_pca(feat, valid=None):
    B, C, D, W = list(feat.size())
    pca, _ = reduce_emb(feat, valid=valid, inbound=None, together=True)
    return pca


def gif_and_tile(ims, just_gif=False):
    S = len(ims)
    # each im is B x H x W x C
    # i want a gif in the left, and the tiled frames on the right
    # for the gif tool, this means making a B x S x H x W tensor
    # where the leftmost part is sequential and the rest is tiled
    gif = torch.stack(ims, dim=1)
    if just_gif:
        return gif
    til = torch.cat(ims, dim=2)
    til = til.unsqueeze(dim=1).repeat(1, S, 1, 1, 1)
    im = torch.cat([gif, til], dim=3)
    return im


def preprocess_color(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32) * 1.0 / 255 - 0.5
    else:
        return x.float() * 1.0 / 255 - 0.5


def back2color(i, blacken_zeros=False):
    if blacken_zeros:
        const = torch.tensor([-0.5])
        i = torch.where(i == 0.0, const.cuda() if i.is_cuda else const, i)
        return back2color(i)
    else:
        return ((i + 0.5) * 255).type(torch.ByteTensor)


def draw_frame_id_on_vis(vis, frame_id, scale=0.5, left=5, top=20, shadow=True):

    rgb = vis.detach().cpu().numpy()[0]
    rgb = np.transpose(rgb, [1, 2, 0])  # put channels last
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    color = (255, 255, 255)
    # print('putting frame id', frame_id)

    frame_str = strnum(frame_id)

    text_color_bg = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(frame_str, font, scale, 1)
    text_w, text_h = text_size
    if shadow:
        cv2.rectangle(
            rgb, (left, top - text_h), (left + text_w, top + 1), text_color_bg, -1
        )

    cv2.putText(
        rgb,
        frame_str,
        (left, top),  # from left, from top
        font,
        scale,  # font scale (float)
        color,
        1,
    )  # font thickness (int)
    rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
    vis = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return vis


def draw_frame_str_on_vis(vis, frame_str, scale=0.5, left=5, top=40, shadow=True):

    rgb = vis.detach().cpu().numpy()[0]
    rgb = np.transpose(rgb, [1, 2, 0])  # put channels last
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    color = (255, 255, 255)

    text_color_bg = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(frame_str, font, scale, 1)
    text_w, text_h = text_size
    if shadow:
        cv2.rectangle(
            rgb, (left, top - text_h), (left + text_w, top + 1), text_color_bg, -1
        )

    cv2.putText(
        rgb,
        frame_str,
        (left, top),  # from left, from top
        font,
        scale,  # font scale (float)
        color,
        1,
    )  # font thickness (int)
    rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
    vis = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return vis


class Summ_writer(object):
    def __init__(
        self, writer, global_step, log_freq=10, fps=8, scalar_freq=100, just_gif=False
    ):
        self.writer = writer
        self.global_step = global_step
        self.log_freq = log_freq
        self.scalar_freq = scalar_freq
        self.fps = fps
        self.just_gif = just_gif
        self.maxwidth = 10000
        self.save_this = self.global_step % self.log_freq == 0
        self.scalar_freq = max(scalar_freq, 1)
        self.save_scalar = self.global_step % self.scalar_freq == 0
        if self.save_this:
            self.save_scalar = True

    def summ_gif(self, name, tensor, blacken_zeros=False):
        # tensor should be in B x S x C x H x W

        assert tensor.dtype in {torch.uint8, torch.float32}
        shape = list(tensor.shape)

        if tensor.dtype == torch.float32:
            tensor = back2color(tensor, blacken_zeros=blacken_zeros)

        video_to_write = tensor[0:1]

        S = video_to_write.shape[1]
        if S == 1:
            # video_to_write is 1 x 1 x C x H x W
            self.writer.add_image(
                name, video_to_write[0, 0], global_step=self.global_step
            )
        else:
            self.writer.add_video(
                name, video_to_write, fps=self.fps, global_step=self.global_step
            )

        return video_to_write

    def summ_rgbs(
        self,
        name,
        ims,
        frame_ids=None,
        frame_strs=None,
        blacken_zeros=False,
        only_return=False,
    ):
        if self.save_this:

            ims = gif_and_tile(ims, just_gif=self.just_gif)
            vis = ims

            assert vis.dtype in {torch.uint8, torch.float32}

            if vis.dtype == torch.float32:
                vis = back2color(vis, blacken_zeros)

            B, S, C, H, W = list(vis.shape)

            if frame_ids is not None:
                assert len(frame_ids) == S
                for s in range(S):
                    vis[:, s] = draw_frame_id_on_vis(vis[:, s], frame_ids[s])

            if frame_strs is not None:
                assert len(frame_strs) == S
                for s in range(S):
                    vis[:, s] = draw_frame_str_on_vis(vis[:, s], frame_strs[s])

            if int(W) > self.maxwidth:
                vis = vis[:, :, :, : self.maxwidth]

            if only_return:
                return vis
            else:
                return self.summ_gif(name, vis, blacken_zeros)

    def summ_rgb(
        self,
        name,
        ims,
        blacken_zeros=False,
        frame_id=None,
        frame_str=None,
        only_return=False,
        halfres=False,
        shadow=True,
    ):
        if self.save_this:
            assert ims.dtype in {torch.uint8, torch.float32}

            if ims.dtype == torch.float32:
                ims = back2color(ims, blacken_zeros)

            # ims is B x C x H x W
            vis = ims[0:1]  # just the first one
            B, C, H, W = list(vis.shape)

            if halfres:
                vis = F.interpolate(vis, scale_factor=0.5)

            if frame_id is not None:
                vis = draw_frame_id_on_vis(vis, frame_id, shadow=shadow)

            if frame_str is not None:
                vis = draw_frame_str_on_vis(vis, frame_str, shadow=shadow)

            if int(W) > self.maxwidth:
                vis = vis[:, :, :, : self.maxwidth]

            if only_return:
                return vis
            else:
                return self.summ_gif(name, vis.unsqueeze(1), blacken_zeros)

    def flow2color(self, flow, clip=0.0):
        B, C, H, W = list(flow.size())
        assert C == 2
        flow = flow[0:1].detach()
        # flow_abs = torch.abs(flow)
        # flow_mean = flow_abs.mean(dim=[1,2,3])
        # flow_std = flow_abs.std(dim=[1,2,3])
        if clip == 0:
            clip = torch.max(torch.abs(flow)).item()

        # if clip:
        flow = torch.clamp(flow, -clip, clip) / clip
        # else:
        #     # # Apply some kind of normalization. Divide by the perceived maximum (mean + std*2)
        #     # flow_max = flow_mean + flow_std*2 + 1e-10
        #     # for b in range(B):
        #     #     flow[b] = flow[b].clamp(-flow_max[b].item(), flow_max[b].item()) / flow_max[b].clamp(min=1)

        #     flow_max = torch.max(flow_abs[b])
        #     for b in range(B):
        #         flow[b] = flow[b].clamp(-flow_max.item(), flow_max.item()) / flow_max[b].clamp(min=1)

        radius = torch.sqrt(torch.sum(flow**2, dim=1, keepdim=True))  # B x 1 x H x W
        radius_clipped = torch.clamp(radius, 0.0, 1.0)

        angle = torch.atan2(-flow[:, 1:2], -flow[:, 0:1]) / np.pi  # B x 1 x H x W

        hue = torch.clamp((angle + 1.0) / 2.0, 0.0, 1.0)
        # hue = torch.mod(angle / (2 * np.pi) + 1.0, 1.0)

        saturation = torch.ones_like(hue) * 0.75
        value = radius_clipped
        hsv = torch.cat([hue, saturation, value], dim=1)  # B x 3 x H x W

        # flow = tf.image.hsv_to_rgb(hsv)
        flow = hsv_to_rgb(hsv)
        flow = (flow * 255.0).type(torch.ByteTensor)
        # flow = torch.flip(flow, dims=[1]).clone() # BGR
        return flow

    def summ_flow(
        self,
        name,
        im,
        clip=0.0,
        only_return=False,
        frame_id=None,
        frame_str=None,
        shadow=True,
    ):
        # flow is B x C x D x W
        if self.save_this:
            return self.summ_rgb(
                name,
                self.flow2color(im, clip=clip),
                only_return=only_return,
                frame_id=frame_id,
                frame_str=frame_str,
                shadow=shadow,
            )
        else:
            return None

    def summ_oneds(
        self,
        name,
        ims,
        frame_ids=None,
        frame_strs=None,
        bev=False,
        fro=False,
        logvis=False,
        reduce_max=False,
        max_val=0.0,
        norm=True,
        only_return=False,
        do_colorize=False,
    ):
        if self.save_this:
            if bev:
                B, C, H, _, W = list(ims[0].shape)
                if reduce_max:
                    ims = [torch.max(im, dim=3)[0] for im in ims]
                else:
                    ims = [torch.mean(im, dim=3) for im in ims]
            elif fro:
                B, C, _, H, W = list(ims[0].shape)
                if reduce_max:
                    ims = [torch.max(im, dim=2)[0] for im in ims]
                else:
                    ims = [torch.mean(im, dim=2) for im in ims]

            if len(ims) != 1:  # sequence
                im = gif_and_tile(ims, just_gif=self.just_gif)
            else:
                im = torch.stack(ims, dim=1)  # single frame

            B, S, C, H, W = list(im.shape)

            if logvis and max_val:
                max_val = np.log(max_val)
                im = torch.log(torch.clamp(im, 0) + 1.0)
                im = torch.clamp(im, 0, max_val)
                im = im / max_val
                norm = False
            elif max_val:
                im = torch.clamp(im, 0, max_val)
                im = im / max_val
                norm = False

            if norm:
                # normalize before oned2inferno,
                # so that the ranges are similar within B across S
                im = normalize(im)

            im = im.view(B * S, C, H, W)
            vis = oned2inferno(im, norm=norm, do_colorize=do_colorize)
            vis = vis.view(B, S, 3, H, W)

            if frame_ids is not None:
                assert len(frame_ids) == S
                for s in range(S):
                    vis[:, s] = draw_frame_id_on_vis(vis[:, s], frame_ids[s])

            if frame_strs is not None:
                assert len(frame_strs) == S
                for s in range(S):
                    vis[:, s] = draw_frame_str_on_vis(vis[:, s], frame_strs[s])

            if W > self.maxwidth:
                vis = vis[..., : self.maxwidth]

            if only_return:
                return vis
            else:
                self.summ_gif(name, vis)

    def summ_oned(
        self,
        name,
        im,
        bev=False,
        fro=False,
        logvis=False,
        max_val=0,
        max_along_y=False,
        norm=True,
        frame_id=None,
        frame_str=None,
        only_return=False,
        shadow=True,
    ):
        if self.save_this:

            if bev:
                B, C, H, _, W = list(im.shape)
                if max_along_y:
                    im = torch.max(im, dim=3)[0]
                else:
                    im = torch.mean(im, dim=3)
            elif fro:
                B, C, _, H, W = list(im.shape)
                if max_along_y:
                    im = torch.max(im, dim=2)[0]
                else:
                    im = torch.mean(im, dim=2)
            else:
                B, C, H, W = list(im.shape)

            im = im[0:1]  # just the first one
            assert C == 1

            if logvis and max_val:
                max_val = np.log(max_val)
                im = torch.log(im)
                im = torch.clamp(im, 0, max_val)
                im = im / max_val
                norm = False
            elif max_val:
                im = torch.clamp(im, 0, max_val) / max_val
                norm = False

            vis = oned2inferno(im, norm=norm)
            if W > self.maxwidth:
                vis = vis[..., : self.maxwidth]
            return self.summ_rgb(
                name,
                vis,
                blacken_zeros=False,
                frame_id=frame_id,
                frame_str=frame_str,
                only_return=only_return,
                shadow=shadow,
            )

    def summ_feats(
        self,
        name,
        feats,
        valids=None,
        pca=True,
        fro=False,
        only_return=False,
        frame_ids=None,
        frame_strs=None,
    ):
        if self.save_this:
            if valids is not None:
                valids = torch.stack(valids, dim=1)

            feats = torch.stack(feats, dim=1)
            # feats leads with B x S x C

            if feats.ndim == 6:

                # feats is B x S x C x D x H x W
                if fro:
                    reduce_dim = 3
                else:
                    reduce_dim = 4

                if valids is None:
                    feats = torch.mean(feats, dim=reduce_dim)
                else:
                    valids = valids.repeat(1, 1, feats.size()[2], 1, 1, 1)
                    feats = reduce_masked_mean(feats, valids, dim=reduce_dim)

            B, S, C, D, W = list(feats.size())

            if not pca:
                # feats leads with B x S x C
                feats = torch.mean(torch.abs(feats), dim=2, keepdims=True)
                # feats leads with B x S x 1
                feats = torch.unbind(feats, dim=1)
                return self.summ_oneds(
                    name=name,
                    ims=feats,
                    norm=True,
                    only_return=only_return,
                    frame_ids=frame_ids,
                    frame_strs=frame_strs,
                )

            else:
                __p = lambda x: pack_seqdim(x, B)
                __u = lambda x: unpack_seqdim(x, B)

                feats_ = __p(feats)

                if valids is None:
                    feats_pca_ = get_feat_pca(feats_)
                else:
                    valids_ = __p(valids)
                    feats_pca_ = get_feat_pca(feats_, valids)

                feats_pca = __u(feats_pca_)

                return self.summ_rgbs(
                    name=name,
                    ims=torch.unbind(feats_pca, dim=1),
                    only_return=only_return,
                    frame_ids=frame_ids,
                    frame_strs=frame_strs,
                )

    def summ_feat(
        self,
        name,
        feat,
        valid=None,
        pca=True,
        only_return=False,
        bev=False,
        fro=False,
        frame_id=None,
        frame_str=None,
    ):
        if self.save_this:
            if feat.ndim == 5:  # B x C x D x H x W

                if bev:
                    reduce_axis = 3
                elif fro:
                    reduce_axis = 2
                else:
                    # default to bev
                    reduce_axis = 3

                if valid is None:
                    feat = torch.mean(feat, dim=reduce_axis)
                else:
                    valid = valid.repeat(1, feat.size()[1], 1, 1, 1)
                    feat = reduce_masked_mean(feat, valid, dim=reduce_axis)

            B, C, D, W = list(feat.shape)

            if not pca:
                feat = torch.mean(torch.abs(feat), dim=1, keepdims=True)
                # feat is B x 1 x D x W
                return self.summ_oned(
                    name=name,
                    im=feat,
                    norm=True,
                    only_return=only_return,
                    frame_id=frame_id,
                    frame_str=frame_str,
                )
            else:
                feat_pca = get_feat_pca(feat, valid)
                return self.summ_rgb(
                    name,
                    feat_pca,
                    only_return=only_return,
                    frame_id=frame_id,
                    frame_str=frame_str,
                )

    def summ_scalar(self, name, value):
        if (
            not (
                isinstance(value, int)
                or isinstance(value, float)
                or isinstance(value, np.float32)
            )
        ) and ("torch" in value.type()):
            value = value.detach().cpu().numpy()
        if not np.isnan(value):
            if self.log_freq == 1:
                self.writer.add_scalar(name, value, global_step=self.global_step)
            elif self.save_this or self.save_scalar:
                self.writer.add_scalar(name, value, global_step=self.global_step)

    def summ_traj2ds_on_rgbs(
        self,
        name,
        trajs,
        rgbs,
        visibs=None,
        valids=None,
        frame_ids=None,
        frame_strs=None,
        only_return=False,
        show_dots=True,
        cmap="coolwarm",
        vals=None,
        linewidth=1,
        max_show=1024,
    ):
        # trajs is B, S, N, 2
        # rgbs is B, S, C, H, W
        B, S, C, H, W = rgbs.shape
        B, S2, N, D = trajs.shape
        assert S == S2

        rgbs = rgbs[0]  # S, C, H, W
        trajs = trajs[0]  # S, N, 2
        if valids is None:
            valids = torch.ones_like(trajs[:, :, 0])  # S, N
        else:
            valids = valids[0]

        if visibs is None:
            visibs = torch.ones_like(trajs[:, :, 0])  # S, N
        else:
            visibs = visibs[0]

        if vals is not None:
            vals = vals[0]  # N
            # print('vals', vals.shape)

        if N > max_show:
            inds = np.random.choice(N, max_show)
            trajs = trajs[:, inds]
            valids = valids[:, inds]
            visibs = visibs[:, inds]
            if vals is not None:
                vals = vals[inds]
            N = trajs.shape[1]

        trajs = trajs.clamp(-16, W + 16)

        rgbs_color = []
        for rgb in rgbs:
            rgb = back2color(rgb).detach().cpu().numpy()
            rgb = np.transpose(rgb, [1, 2, 0])  # put channels last
            rgbs_color.append(rgb)  # each element 3 x H x W

        for i in range(min(N, max_show)):
            if cmap == "onediff" and i == 0:
                cmap_ = "spring"
            elif cmap == "onediff":
                cmap_ = "winter"
            else:
                cmap_ = cmap
            traj = trajs[:, i].long().detach().cpu().numpy()  # S, 2
            valid = valids[:, i].long().detach().cpu().numpy()  # S

            # print('traj', traj.shape)
            # print('valid', valid.shape)

            if vals is not None:
                # val = vals[:,i].float().detach().cpu().numpy() # []
                val = vals[i].float().detach().cpu().numpy()  # []
                # print('val', val.shape)
            else:
                val = None

            for t in range(S):
                if valid[t]:
                    rgbs_color[t] = self.draw_traj_on_image_py(
                        rgbs_color[t],
                        traj[: t + 1],
                        S=S,
                        show_dots=show_dots,
                        cmap=cmap_,
                        val=val,
                        linewidth=linewidth,
                    )

        for i in range(min(N, max_show)):
            if cmap == "onediff" and i == 0:
                cmap_ = "spring"
            elif cmap == "onediff":
                cmap_ = "winter"
            else:
                cmap_ = cmap
            traj = trajs[:, i]  # S,2
            vis = visibs[:, i].round()  # S
            valid = valids[:, i]  # S
            rgbs_color = self.draw_circ_on_images_py(
                rgbs_color,
                traj,
                vis,
                S=S,
                show_dots=show_dots,
                cmap=cmap_,
                linewidth=linewidth,
            )

        rgbs = []
        for rgb in rgbs_color:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            rgbs.append(preprocess_color(rgb))

        return self.summ_rgbs(
            name,
            rgbs,
            only_return=only_return,
            frame_ids=frame_ids,
            frame_strs=frame_strs,
        )

    def summ_traj2ds_on_rgbs2(
        self,
        name,
        trajs,
        visibles,
        rgbs,
        valids=None,
        frame_ids=None,
        frame_strs=None,
        only_return=False,
        show_dots=True,
        cmap=None,
        linewidth=1,
        max_show=1024,
    ):
        # trajs is B, S, N, 2
        # rgbs is B, S, C, H, W
        B, S, C, H, W = rgbs.shape
        B, S2, N, D = trajs.shape
        assert S == S2

        rgbs = rgbs[0]  # S, C, H, W
        trajs = trajs[0]  # S, N, 2
        visibles = visibles[0]  # S, N
        if valids is None:
            valids = torch.ones_like(trajs[:, :, 0])  # S, N
        else:
            valids = valids[0]

        rgbs_color = []
        for rgb in rgbs:
            rgb = back2color(rgb).detach().cpu().numpy()
            rgb = np.transpose(rgb, [1, 2, 0])  # put channels last
            rgbs_color.append(rgb)  # each element 3 x H x W

        trajs = trajs.long().detach().cpu().numpy()  # S, N, 2
        visibles = visibles.float().detach().cpu().numpy()  # S, N
        valids = valids.long().detach().cpu().numpy()  # S, N

        for i in range(min(N, max_show)):
            if cmap == "onediff" and i == 0:
                cmap_ = "spring"
            elif cmap == "onediff":
                cmap_ = "winter"
            else:
                cmap_ = cmap
            traj = trajs[:, i]  # S,2
            vis = visibles[:, i]  # S
            valid = valids[:, i]  # S
            rgbs_color = self.draw_traj_on_images_py(
                rgbs_color,
                traj,
                S=S,
                show_dots=show_dots,
                cmap=cmap_,
                linewidth=linewidth,
            )

        for i in range(min(N, max_show)):
            if cmap == "onediff" and i == 0:
                cmap_ = "spring"
            elif cmap == "onediff":
                cmap_ = "winter"
            else:
                cmap_ = cmap
            traj = trajs[:, i]  # S,2
            vis = visibles[:, i]  # S
            valid = valids[:, i]  # S
            rgbs_color = self.draw_circ_on_images_py(
                rgbs_color,
                traj,
                vis,
                S=S,
                show_dots=show_dots,
                cmap=None,
                linewidth=linewidth,
            )

        rgbs = []
        for rgb in rgbs_color:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            rgbs.append(preprocess_color(rgb))

        return self.summ_rgbs(
            name,
            rgbs,
            only_return=only_return,
            frame_ids=frame_ids,
            frame_strs=frame_strs,
        )

    def summ_traj2ds_on_rgb(
        self,
        name,
        trajs,
        rgb,
        valids=None,
        show_dots=True,
        show_lines=True,
        frame_id=None,
        frame_str=None,
        only_return=False,
        cmap="coolwarm",
        linewidth=1,
        max_show=1024,
    ):
        # trajs is B, S, N, 2
        # rgb is B, C, H, W
        B, C, H, W = rgb.shape
        B, S, N, D = trajs.shape

        rgb = rgb[0]  # S, C, H, W
        trajs = trajs[0]  # S, N, 2

        if valids is None:
            valids = torch.ones_like(trajs[:, :, 0])
        else:
            valids = valids[0]

        rgb_color = back2color(rgb).detach().cpu().numpy()
        rgb_color = np.transpose(rgb_color, [1, 2, 0])  # put channels last

        # using maxdist will dampen the colors for short motions
        # norms = torch.sqrt(1e-4 + torch.sum((trajs[-1] - trajs[0])**2, dim=1)) # N
        # maxdist = torch.quantile(norms, 0.95).detach().cpu().numpy()
        maxdist = None
        trajs = trajs.long().detach().cpu().numpy()  # S, N, 2
        valids = valids.long().detach().cpu().numpy()  # S, N

        if N > max_show:
            inds = np.random.choice(N, max_show)
            trajs = trajs[:, inds]
            valids = valids[:, inds]
            N = trajs.shape[1]

        for i in range(min(N, max_show)):
            if cmap == "onediff" and i == 0:
                cmap_ = "spring"
            elif cmap == "onediff":
                cmap_ = "winter"
            else:
                cmap_ = cmap
            traj = trajs[:, i]  # S, 2
            valid = valids[:, i]  # S
            if valid[0] == 1:
                traj = traj[valid > 0]
                rgb_color = self.draw_traj_on_image_py(
                    rgb_color,
                    traj,
                    S=S,
                    show_dots=show_dots,
                    show_lines=show_lines,
                    cmap=cmap_,
                    maxdist=maxdist,
                    linewidth=linewidth,
                )

        rgb_color = torch.from_numpy(rgb_color).permute(2, 0, 1).unsqueeze(0)
        rgb = preprocess_color(rgb_color)
        return self.summ_rgb(
            name, rgb, only_return=only_return, frame_id=frame_id, frame_str=frame_str
        )

    def draw_traj_on_image_py(
        self,
        rgb,
        traj,
        S=50,
        linewidth=1,
        show_dots=False,
        show_lines=True,
        cmap="coolwarm",
        val=None,
        maxdist=None,
    ):
        # all inputs are numpy tensors
        # rgb is 3 x H x W
        # traj is S x 2

        H, W, C = rgb.shape
        assert C == 3

        rgb = rgb.astype(np.uint8).copy()

        S1, D = traj.shape
        assert D == 2

        color_map = cm.get_cmap(cmap)
        S1, D = traj.shape

        for s in range(S1):
            if val is not None:
                color = np.array(color_map(val)[:3]) * 255  # rgb
            else:
                if maxdist is not None:
                    val = (np.sqrt(np.sum((traj[s] - traj[0]) ** 2)) / maxdist).clip(
                        0, 1
                    )
                    color = np.array(color_map(val)[:3]) * 255  # rgb
                else:
                    color = (
                        np.array(color_map((s) / max(1, float(S - 2)))[:3]) * 255
                    )  # rgb

            if show_lines and s < (S1 - 1):
                cv2.line(
                    rgb,
                    (int(traj[s, 0]), int(traj[s, 1])),
                    (int(traj[s + 1, 0]), int(traj[s + 1, 1])),
                    color,
                    linewidth,
                    cv2.LINE_AA,
                )
            if show_dots:
                cv2.circle(
                    rgb, (int(traj[s, 0]), int(traj[s, 1])), linewidth, color, -1
                )

        # if maxdist is not None:
        #     val = (np.sqrt(np.sum((traj[-1]-traj[0])**2))/maxdist).clip(0,1)
        #     color = np.array(color_map(val)[:3]) * 255 # rgb
        # else:
        #     # draw the endpoint of traj, using the next color (which may be the last color)
        #     color = np.array(color_map((S1-1)/max(1,float(S-2)))[:3]) * 255 # rgb

        # # emphasize endpoint
        # cv2.circle(rgb, (traj[-1,0], traj[-1,1]), linewidth*2, color, -1)

        return rgb

    def draw_traj_on_images_py(
        self,
        rgbs,
        traj,
        S=50,
        linewidth=1,
        show_dots=False,
        cmap="coolwarm",
        maxdist=None,
    ):
        # all inputs are numpy tensors
        # rgbs is a list of H,W,3
        # traj is S,2
        H, W, C = rgbs[0].shape
        assert C == 3

        rgbs = [rgb.astype(np.uint8).copy() for rgb in rgbs]

        S1, D = traj.shape
        assert D == 2

        x = int(np.clip(traj[0, 0], 0, W - 1))
        y = int(np.clip(traj[0, 1], 0, H - 1))
        color = rgbs[0][y, x]
        color = (int(color[0]), int(color[1]), int(color[2]))
        for s in range(S):
            # bak_color = np.array(color_map(1.0)[:3]) * 255 # rgb
            # cv2.circle(rgbs[s], (traj[s,0], traj[s,1]), linewidth*4, bak_color, -1)
            cv2.polylines(
                rgbs[s], [traj[: s + 1]], False, color, linewidth, cv2.LINE_AA
            )
        return rgbs

    def draw_circs_on_image_py(
        self,
        rgb,
        xy,
        colors=None,
        linewidth=10,
        radius=3,
        show_dots=False,
        maxdist=None,
    ):
        # all inputs are numpy tensors
        # rgbs is a list of 3,H,W
        # xy is N,2
        H, W, C = rgb.shape
        assert C == 3

        rgb = rgb.astype(np.uint8).copy()

        N, D = xy.shape
        assert D == 2

        xy = xy.astype(np.float32)
        xy[:, 0] = np.clip(xy[:, 0], 0, W - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, H - 1)
        xy = xy.astype(np.int32)

        if colors is None:
            colors = get_n_colors(N)

        for n in range(N):
            color = colors[n]
            # print('color', color)
            # color = (color[0]*255).astype(np.uint8)
            color = (int(color[0]), int(color[1]), int(color[2]))

            # x = int(np.clip(xy[0,0], 0, W-1))
            # y = int(np.clip(xy[0,1], 0, H-1))
            # color_ = rgbs[0][y,x]
            # color_ = (int(color_[0]),int(color_[1]),int(color_[2]))
            # color_ = (int(color_[0]),int(color_[1]),int(color_[2]))

            cv2.circle(rgb, (int(xy[n, 0]), int(xy[n, 1])), linewidth, color, 3)
            # vis_color = int(np.squeeze(vis[s])*255)
            # vis_color = (vis_color,vis_color,vis_color)
            # cv2.circle(rgbs[s], (traj[s,0], traj[s,1]), linewidth+1, vis_color, -1)
        return rgb

    def draw_circ_on_images_py(
        self,
        rgbs,
        traj,
        vis,
        S=50,
        linewidth=1,
        show_dots=False,
        cmap=None,
        maxdist=None,
    ):
        # all inputs are numpy tensors
        # rgbs is a list of 3,H,W
        # traj is S,2
        H, W, C = rgbs[0].shape
        assert C == 3

        rgbs = [rgb.astype(np.uint8).copy() for rgb in rgbs]

        S1, D = traj.shape
        assert D == 2

        if cmap is None:
            bremm = ColorMap2d()
            traj_ = traj[0:1].astype(np.float32)
            traj_[:, 0] /= float(W)
            traj_[:, 1] /= float(H)
            color = bremm(traj_)
            # print('color', color)
            color = (color[0] * 255).astype(np.uint8)
            color = (int(color[0]), int(color[1]), int(color[2]))

        for s in range(S):
            if cmap is not None:
                color_map = cm.get_cmap(cmap)
                # color = np.array(color_map(s/(S-1))[:3]) * 255 # rgb
                color = np.array(color_map((s) / max(1, float(S - 2)))[:3]) * 255  # rgb
                # color = color.astype(np.uint8)
                # color = (color[0], color[1], color[2])
                # print('color', color)
            # import ipdb; ipdb.set_trace()

            cv2.circle(
                rgbs[s], (int(traj[s, 0]), int(traj[s, 1])), linewidth + 2, color, -1
            )
            vis_color = int(np.squeeze(vis[s]) * 255)
            vis_color = (vis_color, vis_color, vis_color)
            cv2.circle(
                rgbs[s],
                (int(traj[s, 0]), int(traj[s, 1])),
                linewidth + 1,
                vis_color,
                -1,
            )

        return rgbs

    def summ_pts_on_rgb(
        self,
        name,
        trajs,
        rgb,
        visibs=None,
        valids=None,
        frame_id=None,
        frame_str=None,
        only_return=False,
        show_dots=True,
        colors=None,
        cmap="coolwarm",
        linewidth=1,
        max_show=1024,
        already_sorted=False,
    ):
        # trajs is B, S, N, 2
        # rgbs is B, S, C, H, W
        B, C, H, W = rgb.shape
        B, S, N, D = trajs.shape

        rgb = rgb[0]  # C, H, W
        trajs = trajs[0]  # S, N, 2
        if valids is None:
            valids = torch.ones_like(trajs[:, :, 0])  # S, N
        else:
            valids = valids[0]
        if visibs is None:
            visibs = torch.ones_like(trajs[:, :, 0])  # S, N
        else:
            visibs = visibs[0]

        trajs = trajs.clamp(-16, W + 16)

        if N > max_show:
            inds = np.random.choice(N, max_show)
            trajs = trajs[:, inds]
            valids = valids[:, inds]
            visibs = visibs[:, inds]
            N = trajs.shape[1]

        if not already_sorted:
            inds = torch.argsort(torch.mean(trajs[:, :, 1], dim=0))
            trajs = trajs[:, inds]
            valids = valids[:, inds]
            visibs = visibs[:, inds]

        rgb = back2color(rgb).detach().cpu().numpy()
        rgb = np.transpose(rgb, [1, 2, 0])  # put channels last

        trajs = trajs.long().detach().cpu().numpy()  # S, N, 2
        valids = valids.long().detach().cpu().numpy()  # S, N
        visibs = visibs.long().detach().cpu().numpy()  # S, N

        rgb = rgb.astype(np.uint8).copy()

        for i in range(min(N, max_show)):
            if cmap == "onediff" and i == 0:
                cmap_ = "spring"
            elif cmap == "onediff":
                cmap_ = "winter"
            else:
                cmap_ = cmap
            traj = trajs[:, i]  # S,2
            valid = valids[:, i]  # S
            visib = visibs[:, i]  # S

            if colors is None:
                ii = i / (1e-4 + N - 1.0)
                color_map = cm.get_cmap(cmap)
                color = np.array(color_map(ii)[:3]) * 255  # rgb
            else:
                color = np.array(colors[i]).astype(np.int64)
            color = (int(color[0]), int(color[1]), int(color[2]))

            for s in range(S):
                if valid[s]:
                    if visib[s]:
                        thickness = -1
                    else:
                        thickness = 2
                    cv2.circle(
                        rgb,
                        (int(traj[s, 0]), int(traj[s, 1])),
                        linewidth,
                        color,
                        thickness,
                    )
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        rgb = preprocess_color(rgb)
        return self.summ_rgb(
            name, rgb, only_return=only_return, frame_id=frame_id, frame_str=frame_str
        )

    def summ_pts_on_rgbs(
        self,
        name,
        trajs,
        rgbs,
        visibs=None,
        valids=None,
        frame_ids=None,
        only_return=False,
        show_dots=True,
        cmap="coolwarm",
        colors=None,
        linewidth=1,
        max_show=1024,
        frame_strs=None,
    ):
        # trajs is B, S, N, 2
        # rgbs is B, S, C, H, W
        B, S, C, H, W = rgbs.shape
        B, S2, N, D = trajs.shape
        assert S == S2

        rgbs = rgbs[0]  # S, C, H, W
        trajs = trajs[0]  # S, N, 2
        if valids is None:
            valids = torch.ones_like(trajs[:, :, 0])  # S, N
        else:
            valids = valids[0]
        if visibs is None:
            visibs = torch.ones_like(trajs[:, :, 0])  # S, N
        else:
            visibs = visibs[0]

        if N > max_show:
            inds = np.random.choice(N, max_show)
            trajs = trajs[:, inds]
            valids = valids[:, inds]
            visibs = visibs[:, inds]
            N = trajs.shape[1]
        inds = torch.argsort(torch.mean(trajs[:, :, 1], dim=0))
        trajs = trajs[:, inds]
        valids = valids[:, inds]
        visibs = visibs[:, inds]

        rgbs_color = []
        for rgb in rgbs:
            rgb = back2color(rgb).detach().cpu().numpy()
            rgb = np.transpose(rgb, [1, 2, 0])  # put channels last
            rgbs_color.append(rgb)  # each element 3 x H x W

        trajs = trajs.long().detach().cpu().numpy()  # S, N, 2
        valids = valids.long().detach().cpu().numpy()  # S, N
        visibs = visibs.long().detach().cpu().numpy()  # S, N

        rgbs_color = [rgb.astype(np.uint8).copy() for rgb in rgbs_color]

        for i in range(min(N, max_show)):
            traj = trajs[:, i]  # S,2
            valid = valids[:, i]  # S
            visib = visibs[:, i]  # S

            if colors is None:
                ii = i / (1e-4 + N - 1.0)
                color_map = cm.get_cmap(cmap)
                color = np.array(color_map(ii)[:3]) * 255  # rgb
            else:
                color = np.array(colors[i]).astype(np.int64)
            color = (int(color[0]), int(color[1]), int(color[2]))

            for s in range(S):
                if valid[s]:
                    if visib[s]:
                        thickness = -1
                    else:
                        thickness = 2
                    cv2.circle(
                        rgbs_color[s],
                        (int(traj[s, 0]), int(traj[s, 1])),
                        int(linewidth),
                        color,
                        thickness,
                    )
        rgbs = []
        for rgb in rgbs_color:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            rgbs.append(preprocess_color(rgb))

        return self.summ_rgbs(
            name,
            rgbs,
            only_return=only_return,
            frame_ids=frame_ids,
            frame_strs=frame_strs,
        )
