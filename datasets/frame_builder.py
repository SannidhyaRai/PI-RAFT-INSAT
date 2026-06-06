import torch

def build_temporal_sequence(frames_list, sequence_length=2):
    """
    Builds temporal sequences from a list of loaded frames.
    Currently defaults to returning 2 consecutive frames.
    
    Args:
        frames_list (list): List of loaded frame tensors.
        sequence_length (int): Target temporal frame length (e.g., 2 or 4).
    """
    # TODO: scientific validation required
    # Deferred from audit phase: 4-frame tracking workflow implementation
    if sequence_length == 2:
        # Returns consecutive frame pairs
        return [(frames_list[i], frames_list[i+1]) for i in range(len(frames_list) - 1)]
    elif sequence_length == 4:
        # Placeholder for 4-frame sequences
        # Returns [t0, t1, t2, t3] groupings
        sequences = []
        for i in range(len(frames_list) - 3):
            sequences.append(frames_list[i : i+4])
        return sequences
    else:
        raise ValueError(f"Unsupported sequence length: {sequence_length}")
