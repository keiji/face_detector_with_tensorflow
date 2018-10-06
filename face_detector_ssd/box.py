class Box:
  label = [0]
  label_mask = 0
  offset = [0.0, 0.0, 0.0, 0.0]

  def __init__(self, left, top, right, bottom):
    self.left = left
    self.top = top
    self.right = right
    self.bottom = bottom

  def __repr__(self):
    return '[%f, %f, %f, %f]' % (self.left, self.top, self.right, self.bottom)
