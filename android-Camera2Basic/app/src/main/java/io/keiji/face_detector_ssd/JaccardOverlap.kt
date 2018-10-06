package io.keiji.face_detector_ssd

import android.graphics.RectF

private fun land(rect: RectF): Float {
    val width = rect.right - rect.left
    val height = rect.bottom - rect.top
    if (width < 0 || height < 0) {
        return 0.0F
    }
    return width * height
}

fun jaccard_overlap(lRect: RectF, rRect: RectF): Float {
    val overlapLeft = maxOf(lRect.left, rRect.left)
    val overlapTop = maxOf(lRect.top, rRect.top)
    val overlapRight = minOf(lRect.right, rRect.right)
    val overlapBottom = minOf(lRect.bottom, rRect.bottom)

    val overlapRect = RectF(overlapLeft, overlapTop, overlapRight, overlapBottom)

    val overlapLand = land(overlapRect)
    val unionLand = land(lRect) + land(rRect) - overlapLand

    return overlapLand / unionLand
}
