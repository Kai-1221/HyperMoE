"""Bottom-up context propagation over the five patient feature levels."""
import torch
import torch.nn.functional as F


def build_unified_topology(wsi_patches, wsi_regions, wsi_slide,
                           mri_roi=None, mri_fs=None, p2r=None, guide=None):
    has_wsi = wsi_patches is not None
    has_mri = mri_roi is not None
    if not has_wsi and not has_mri:
        raise ValueError("At least one complete modality is required")

    parts = []
    if has_wsi:
        if wsi_regions is None or wsi_slide is None or p2r is None:
            raise ValueError("WSI requires patches, regions, slide and patch-to-region map")
        n, m = len(wsi_patches), len(wsi_regions)
        if n == 0 or m == 0 or len(wsi_slide) != 1 or p2r.numel() != n:
            raise ValueError("Invalid WSI hierarchy shapes")
        p2r = p2r.reshape(-1).long()
        if (p2r < 0).any() or (p2r >= m).any():
            raise ValueError("patch-to-region indices are out of range")
        parts.extend((wsi_patches, wsi_regions, wsi_slide))
    else:
        n = m = 0
    if has_mri:
        if mri_fs is None or len(mri_roi) != 1 or len(mri_fs) != 1:
            raise ValueError("MRI requires one ROI vector and one global vector")
        parts.extend((mri_roi, mri_fs))

    nodes = torch.cat(parts, dim=0)
    if has_mri:
        root = mri_fs if guide is None else guide
        similarity = (1.0 + F.cosine_similarity(nodes, root, dim=-1)) / 2.0
        weighted = nodes * similarity.unsqueeze(-1)
    else:
        weighted = nodes

    output = weighted.clone()
    if has_wsi:
        slide_index = n + m
        patch_means = weighted.new_zeros((m, weighted.size(-1)))
        counts = weighted.new_zeros((m, 1))
        patch_means.index_add_(0, p2r, weighted[:n])
        counts.index_add_(0, p2r, weighted.new_ones((n, 1)))
        patch_means = patch_means / counts.clamp_min(1.0)
        region_context = 0.5 * (weighted[n:n + m] + patch_means)
        slide_context = 0.5 * (weighted[slide_index:slide_index + 1] +
                               region_context.mean(0, keepdim=True))
        output[n:n + m] = region_context
        output[slide_index:slide_index + 1] = slide_context
    else:
        slide_context = None

    if has_mri:
        roi_index = len(nodes) - 2
        fs_index = len(nodes) - 1
        roi_context = weighted[roi_index:roi_index + 1]
        if slide_context is not None:
            roi_context = 0.5 * (roi_context + slide_context)
        fs_context = 0.5 * (weighted[fs_index:fs_index + 1] + roi_context)
        output[roi_index:roi_index + 1] = roi_context
        output[fs_index:fs_index + 1] = fs_context

    return output, output[-1:], nodes
