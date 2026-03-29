"""
finite_future_feature_extractor.py - Future (test) feature extractor factory.

Paper Reference: Hefny et al. (2017), arXiv:1702.03537

Creates a feature extractor for future (test) windows:
    q_t = [o_t, o_{t+1}, ..., o_{t+k-1}] (flattened)

With lag > 0, the window starts at t + lag instead of t. 
Out-of-bounds frames are zero-padded.

Paper: Sec. 3.1 - test featres phi(q_t) where q_t is the future window.
"""