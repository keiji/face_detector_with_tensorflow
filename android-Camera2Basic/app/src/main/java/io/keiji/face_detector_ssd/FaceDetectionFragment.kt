package io.keiji.face_detector_ssd

import android.graphics.Bitmap
import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.os.Handler
import android.support.v4.app.Fragment
import android.view.View
import com.example.android.camera2basic.Camera2BasicFragment
import com.example.android.camera2basic.R
import java.nio.ByteBuffer
import java.nio.ByteOrder

class FaceDetectionFragment : Camera2BasicFragment() {

    companion object {
        val TAG = FaceDetectionFragment::class.java.simpleName

        fun newInstance(): Fragment {
            return FaceDetectionFragment()
        }
    }

    val handler = Handler()

    lateinit var ssdFaceDetector: SsdFaceDetector
//    lateinit var ssdFaceDetector: SsdFaceDetectorMobile

    override fun onStart() {
        super.onStart()

        ssdFaceDetector = SsdFaceDetector(context!!.assets, 0.99F)
//        ssdFaceDetector = SsdFaceDetectorMobile(context!!.assets, 0.99F)
    }

    override fun onStop() {
        super.onStop()

        ssdFaceDetector.stop()
    }

    lateinit var faceView: FaceView

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        faceView = view.findViewById(R.id.faces)

    }

    override fun primaryCameraDirection(): Int {
        return CameraCharacteristics.LENS_FACING_FRONT
//        return CameraCharacteristics.LENS_FACING_BACK
    }

    val imageByteBuffer: ByteBuffer =
            ByteBuffer.allocateDirect(SsdFaceDetector.IMAGE_BYTES_LENGTH).also {
                it.order(ByteOrder.nativeOrder())
            }

    override fun onReceivedPreviewImage(bitmap: Bitmap) {
        val resizedImage = SsdFaceDetector.resizeToPreferSize(bitmap)

        synchronized(ssdFaceDetector) {
            resizedImage.copyPixelsToBuffer(imageByteBuffer)
            imageByteBuffer.rewind()

            // https://github.com/CyberAgent/android-gpuimage/issues/24
            if (bitmap != resizedImage) {
                resizedImage.recycle()
            }

            try {
                val faceList = ssdFaceDetector.recognize(imageByteBuffer)

                handler.post {
                    faceView.faceList = faceList
                }
            } finally {
                imageByteBuffer.clear()
            }
        }
    }
}