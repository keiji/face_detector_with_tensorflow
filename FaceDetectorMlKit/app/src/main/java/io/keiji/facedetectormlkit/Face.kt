package io.keiji.facedetectormlkit

import android.graphics.Bitmap
import android.graphics.Rect
import com.google.firebase.ml.vision.common.FirebaseVisionPoint
import com.google.gson.annotations.SerializedName

class Face(
        @SerializedName("file_name")
        val fileName: String,

        @Transient
        val bitmap: Bitmap,
        faceRect: Rect) {

    @SerializedName("position")
    val position: BoundingBox

    init {
        position = BoundingBox(
                faceRect.left.toFloat() / bitmap.width,
                faceRect.top.toFloat() / bitmap.height,
                faceRect.right.toFloat() / bitmap.width,
                faceRect.bottom.toFloat() / bitmap.height
        )
    }

    @SerializedName("left_eye")
    var leftEye: BoundingBox? = null
    fun setLeftEye(position: FirebaseVisionPoint) {
        leftEye = BoundingBox(
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height,
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height
        )
    }

    @SerializedName("right_eye")
    var rightEye: BoundingBox? = null
    fun setRightEye(position: FirebaseVisionPoint) {
        rightEye = BoundingBox(
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height,
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height
        )
    }

    @SerializedName("nose_base")
    var noseBase: BoundingBox? = null
    fun setNoseBase(position: FirebaseVisionPoint) {
        noseBase = BoundingBox(
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height,
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height
        )
    }

    @SerializedName("left_mouth")
    var leftMouth: BoundingBox? = null
    fun setLeftMouth(position: FirebaseVisionPoint) {
        leftMouth = BoundingBox(
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height,
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height
        )
    }

    @SerializedName("right_mouth")
    var rightMouth: BoundingBox? = null
    fun setRightMouth(position: FirebaseVisionPoint) {
        rightMouth = BoundingBox(
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height,
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height
        )
    }

    @SerializedName("bottom_mouth")
    var bottomMouth: BoundingBox? = null
    fun setBottomMouth(position: FirebaseVisionPoint) {
        bottomMouth = BoundingBox(
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height,
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height
        )
    }

    @SerializedName("left_cheek")
    var leftCheek: BoundingBox? = null
    fun setLeftCheek(position: FirebaseVisionPoint) {
        leftCheek = BoundingBox(
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height,
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height
        )
    }

    @SerializedName("right_cheek")
    var rightCheek: BoundingBox? = null
    fun setRightCheek(position: FirebaseVisionPoint) {
        rightCheek = BoundingBox(
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height,
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height
        )
    }

    @SerializedName("left_ear")
    var leftEar: BoundingBox? = null
    fun setLeftEar(position: FirebaseVisionPoint) {
        leftEar = BoundingBox(
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height,
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height
        )
    }

    @SerializedName("right_ear")
    var rightEar: BoundingBox? = null
    fun setRightEar(position: FirebaseVisionPoint) {
        rightEar = BoundingBox(
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height,
                position.x.toFloat() / bitmap.width,
                position.y.toFloat() / bitmap.height
        )
    }

    class BoundingBox(
            @SerializedName("left")
            var left: Float,

            @SerializedName("top")
            var top: Float,

            @SerializedName("right")
            var right: Float,

            @SerializedName("bottom")
            var bottom: Float) {
    }
}