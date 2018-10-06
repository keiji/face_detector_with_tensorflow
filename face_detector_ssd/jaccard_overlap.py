def _land(rect):
  left, top, right, bottom = rect
  width = right - left
  height = bottom - top
  if width < 0 or height < 0:
    return 0
  return width * height


def jaccard_overlap(rect1, rect2):
  rect1_left, rect1_top, rect1_right, rect1_bottom = rect1
  rect2_left, rect2_top, rect2_right, rect2_bottom = rect2

  overlap_left = max(rect1_left, rect2_left)
  overlap_top = max(rect1_top, rect2_top)
  overlap_right = min(rect1_right, rect2_right)
  overlap_bottom = min(rect1_bottom, rect2_bottom)

  overlap = _land([overlap_left, overlap_top, overlap_right, overlap_bottom])
  union = _land(rect1) + _land(rect2) - overlap

  return overlap / union
