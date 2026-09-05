// Shared indexing helpers for tensor buffer layout.
// All tensors in this pipeline use BCHW (Batch, Channel, Height, Width).

// Flat index for element at (c, y, x) in a [C, H, W] plane.
// For batched tensors [B, C, H, W], prepend batch * C * H * W.
uint chw_index(uint H, uint W, uint c, uint y, uint x) {
  return c * (H * W) + y * W + x;
}

// Flat index for element at (y, x, c) in a [H, W, C] plane (BHWC).
// Not used in the current BCHW pipeline; provided for reference.
uint hwc_index(uint W, uint C, uint y, uint x, uint c) {
  return (y * W + x) * C + c;
}
