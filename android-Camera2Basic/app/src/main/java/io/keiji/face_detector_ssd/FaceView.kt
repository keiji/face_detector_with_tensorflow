package io.keiji.face_detector_ssd

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class FaceView @JvmOverloads
constructor(context: Context,
            attrs: AttributeSet? = null,
            defStyle: Int = 0) : View(context, attrs, defStyle) {

    companion object {
        val TAG = FaceView::class.java.simpleName
    }

    var faceList: List<Face> = emptyList()
        set(value) {
            field = value

            invalidate()
        }

    val paint = Paint().also {
        it.color = Color.RED
        it.style = Paint.Style.STROKE
        it.strokeWidth = 3.0F
    }

    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)

        canvas ?: return

        for (face in faceList) {
            if (face.confidence < 0.0F) {
                continue
            }
            val left = face.rect.left * width
            val top = face.rect.top * height
            val right = face.rect.right * width
            val bottom = face.rect.bottom * height

            canvas.drawRect(left, top, right, bottom, paint)
        }
    }
}