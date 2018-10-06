package io.keiji.face_detector_ssd

import android.graphics.RectF

class Face(
        var confidence: Float,
        val rect: RectF = RectF()
)