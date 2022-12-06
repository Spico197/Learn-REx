import torch


def extract_positions_from_start_end_label(
    start_label: torch.Tensor, end_label: torch.Tensor
):
    start_idxes = start_label.eq(1).nonzero(as_tuple=False).reshape(-1).tolist()
    end_idxes = end_label.eq(1).nonzero(as_tuple=False).reshape(-1).tolist()
    positions = []
    for i, start_idx in enumerate(start_idxes):
        next_start = (
            start_idxes[i + 1] if i < len(start_idxes) - 1 else len(start_label)
        )
        for end_idx in end_idxes:
            if end_idx >= start_idx and end_idx < next_start:
                positions.append((start_idx, end_idx))
                break
    return positions
