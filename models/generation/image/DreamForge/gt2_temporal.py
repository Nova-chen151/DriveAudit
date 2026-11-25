"""Native DreamForge reference/overlap indexing for the GT2 future horizon."""


def history_overlap_windows(overlap_length=5, length=7):
    """Continue history0..9, retaining a causal prefix in each fixed-size window."""
    if not 1 <= overlap_length < length or overlap_length > 8:
        raise ValueError('Overlap must leave a new frame and two available history references')
    windows = []
    for first_new in range(10, 90, length - overlap_length):
        start = first_new - overlap_length
        windows.append(dict(
            references=[start - 2, start - 1],
            frames=[min(f, 89) for f in range(start, start + length)],
            overlaps=list(range(start, first_new)),
            outputs=list(range(first_new, min(start + length, 90)))))
    return windows


def overlap_windows(length=7, warm_start=False):
    if length < 3:
        raise ValueError('Need two reference frames before the overlapping frame')
    windows = []
    start = 9 if warm_start else 10
    while start < 90:
        first = not windows and not warm_start
        outputs = list(range(start if first else start + 1, min(start + length, 90)))
        windows.append(dict(
            references=[9, 9] if first else [start - 2, start - 1],
            frames=[min(frame, 89) for frame in range(start, start + length)],
            outputs=outputs, overlap=None if first else start))
        if outputs[-1] == 89:
            break
        start = outputs[-1]
    return windows
