def foot_point(x1, y1, x2, y2):
    """Return the ground-plane contact point (bottom-centre of bounding box)."""
    return (x1 + x2) // 2, y2
