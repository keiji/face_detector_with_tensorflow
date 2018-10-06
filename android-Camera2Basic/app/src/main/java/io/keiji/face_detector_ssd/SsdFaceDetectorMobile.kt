package io.keiji.face_detector_ssd

/*
Copyright 2018 Keiji ARIYAMA

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 */

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Debug
import android.util.Log
import org.json.JSONArray
import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.collections.ArrayList


class SsdFaceDetectorMobile(assetManager: AssetManager,
                            val confidenceThreshold: Float) {

    private class Box() {
        var left: Float = 0.0F
        var top: Float = 0.0F
        var right: Float = 0.0F
        var bottom: Float = 0.0F
    }

    companion object {
        val TAG = SsdFaceDetectorMobile::class.java.simpleName

        private val MODEL_FILE_NAME = "model1_4ch.pb"
        private val BOXES_POSITION_FILE_NAME = "model1_boxes_position.json"

        private val IMAGE_WIDTH = 128
        private val IMAGE_HEIGHT = 128
        private val IMAGE_CHANNEL = 4

        val IMAGE_BYTES_LENGTH = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL

        fun resizeToPreferSize(bitmap: Bitmap): Bitmap {
            return Bitmap.createScaledBitmap(bitmap,
                    IMAGE_WIDTH, IMAGE_HEIGHT, false)
        }
    }

    private val boxes: Array<Box>

    private val resultOffset: FloatArray
    private val resultConfidence: FloatArray

    val tfInference: TensorFlowInferenceInterface = TensorFlowInferenceInterface(
            assetManager.open(MODEL_FILE_NAME))

    private val faces = ArrayList<Face>()

    init {
        boxes = loadBoxes(assetManager, BOXES_POSITION_FILE_NAME)

        resultConfidence = FloatArray(boxes.size)
        resultOffset = FloatArray(boxes.size * 4)
    }

    private fun loadBoxes(assetManager: AssetManager, jsonFileName: String): Array<Box> {
        assetManager.open(jsonFileName).use {
            val jsonArray = JSONArray(InputStreamReader(it).readText())

            val boxes = Array(jsonArray.length(), { Box() })

            for (index in 0..jsonArray.length() - 1) {
                val b = jsonArray.getJSONObject(index)
                boxes[index].also {
                    it.left = b.getDouble("left").toFloat()
                    it.top = b.getDouble("top").toFloat()
                    it.right = b.getDouble("right").toFloat()
                    it.bottom = b.getDouble("bottom").toFloat()
                }
            }

            return boxes
        }
    }

    fun recognize(byteBuffer: ByteBuffer): List<Face> {
        val start = Debug.threadCpuTimeNanos()

        tfInference.feed("input", byteBuffer, IMAGE_BYTES_LENGTH.toLong())
        tfInference.run(arrayOf("confidence", "offset"))
        tfInference.fetch("confidence", resultConfidence)
        tfInference.fetch("offset", resultOffset)

        val inferenceElapsed = (Debug.threadCpuTimeNanos() - start) / 1000000
        Log.d(TAG, "Inference Elapsed: %,3d ms".format(inferenceElapsed))

        boxesToFaces(boxes, faces)
        mergeFaces(faces, overlapThreshold = 0.3F)

        return faces
    }

    private fun boxesToFaces(boxes: Array<SsdFaceDetectorMobile.Box>,
                             faces: ArrayList<Face>) {
        faces.clear()

        for (index in 0..resultConfidence.size - 1) {
            val confidence = resultConfidence[index]
            if (confidence < confidenceThreshold) {
                continue
            }

            val box = boxes[index]
            val i = index * 4

            val offsetLeft = resultOffset[i]
            val offsetTop = resultOffset[i + 1]
            val offsetRight = resultOffset[i + 2]
            val offsetBottom = resultOffset[i + 3]

            faces.add(Face(confidence,
                    RectF(box.left + offsetLeft,
                            box.top + offsetTop,
                            box.right + offsetRight,
                            box.bottom + offsetBottom)
            ))
        }
    }

    private fun mergeFaces(faces: ArrayList<Face>,
                           overlapThreshold: Float = 0.5F) {

        Collections.sort(faces,
                { lFace: Face, rFace: Face ->
                    if (lFace.confidence < rFace.confidence) 1 else -1
                })

        for (baseFace in faces) {
            if (baseFace.confidence < confidenceThreshold) {
                continue
            }

            while (true) {
                var mergedFlag = false

                for (targetFace in faces) {
                    if (targetFace === baseFace
                            || targetFace.confidence < confidenceThreshold) {
                        continue
                    }

                    val overlap = jaccard_overlap(
                            baseFace.rect,
                            targetFace.rect
                    )

                    if (overlap >= overlapThreshold) {
                        mergeFaces(baseFace, targetFace)
                        mergedFlag = true
                    }
                }

                if (!mergedFlag) {
                    break
                }
            }
        }
    }

    private fun mergeFaces(baseFace: Face, targetFace: Face) {
        val baseWeight = baseFace.confidence /
                (baseFace.confidence + targetFace.confidence)
        val targetWeight = 1.0F - baseWeight

        baseFace.rect.left = (baseFace.rect.left * baseWeight) +
                (targetFace.rect.left * targetWeight)
        baseFace.rect.top = (baseFace.rect.top * baseWeight) +
                (targetFace.rect.top * targetWeight)
        baseFace.rect.right = (baseFace.rect.right * baseWeight) +
                (targetFace.rect.right * targetWeight)
        baseFace.rect.bottom = (baseFace.rect.bottom * baseWeight) +
                (targetFace.rect.bottom * targetWeight)

        targetFace.confidence = -1.0F
    }

    fun stop() {
        tfInference.close()
    }
}