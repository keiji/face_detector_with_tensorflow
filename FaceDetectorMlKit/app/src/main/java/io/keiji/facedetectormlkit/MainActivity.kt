package io.keiji.facedetectormlkit

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import com.google.firebase.ml.vision.common.FirebaseVisionImage
import com.google.firebase.ml.vision.face.FirebaseVisionFace
import io.reactivex.Single
import io.reactivex.SingleObserver
import android.graphics.BitmapFactory
import android.os.Debug
import android.support.annotation.NonNull
import android.util.Log
import android.view.WindowManager
import android.widget.ImageView
import android.widget.TextView
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions
import com.google.firebase.ml.vision.FirebaseVision
import com.google.android.gms.tasks.OnFailureListener
import com.google.firebase.ml.vision.face.FirebaseVisionFaceLandmark
import com.google.gson.GsonBuilder
import io.reactivex.android.schedulers.AndroidSchedulers
import io.reactivex.disposables.CompositeDisposable
import io.reactivex.schedulers.Schedulers
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.adapter.rxjava2.RxJava2CallAdapterFactory
import retrofit2.converter.gson.GsonConverterFactory
import java.io.*
import java.util.concurrent.TimeUnit


class MainActivity : AppCompatActivity() {

    companion object {
        val TAG = MainActivity::class.java.simpleName

        val API_BASE_URL = "http://192.168.170.10:3000"
        val LOG_FILE_NAME = "elapsed.log"

        val OPTIONS = FirebaseVisionFaceDetectorOptions.Builder()
                .setModeType(FirebaseVisionFaceDetectorOptions.ACCURATE_MODE)
                .setLandmarkType(FirebaseVisionFaceDetectorOptions.ALL_LANDMARKS)
                .setClassificationType(
                        FirebaseVisionFaceDetectorOptions.NO_CLASSIFICATIONS)
                .setMinFaceSize(0.025f)
                .setTrackingEnabled(false)
                .build()
    }

    private var compositeDisposable: CompositeDisposable = CompositeDisposable()

    val api: Api = Retrofit.Builder()
            .baseUrl(API_BASE_URL)
            .addConverterFactory(GsonConverterFactory.create(GsonBuilder().create()))
            .addCallAdapterFactory(RxJava2CallAdapterFactory.create())
            .client(OkHttpClient()
                    .newBuilder()
                    .readTimeout(10, TimeUnit.SECONDS)
                    .connectTimeout(10, TimeUnit.SECONDS)
                    .build())
            .build()
            .create(Api::class.java)

    val faceDetector = FirebaseVision.getInstance()
            .getVisionFaceDetector(OPTIONS)

    var offset = 0
    var limit = 100

    var processCount = 0
    var totalCount = 0

    private lateinit var textView: TextView
    private lateinit var imageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main)

        textView = findViewById(R.id.textview)
        imageView = findViewById(R.id.imageview)

        logFileOutputStream = OutputStreamWriter(
                openFileOutput(LOG_FILE_NAME, Context.MODE_APPEND))

        nextBatch()
    }

    fun nextBatch() {
        compositeDisposable.add(GetLastOffset(openFileInput(LOG_FILE_NAME))
                .subscribeOn(Schedulers.newThread())
                .flatMap {
                    processCount = it
                    offset = it
                    totalCount = it
                    api.getFileList(limit, offset)
                }
                .map { fileNameList ->
                    if (fileNameList.isEmpty()) {
                        return@map
                    }

                    totalCount += fileNameList.size
                    offset += fileNameList.size

                    val disposable = Single.concat(
                            fileNameList.map { fileName ->
                                api.getBitmap(fileName)
                                        .subscribeOn(Schedulers.newThread())
                                        .flatMap { CreateBitmap(it.byteStream()) }
                                        .flatMap { ResizeBitmap(Container(fileName, it)) }
                                        .flatMap { DetectFace(it) }
                                        .flatMap {
                                            val (container, visionFaceList) = it
                                            ConvertToFaceList(container, visionFaceList)
                                        }
                                        .doOnSuccess { container ->
                                            processCount++

                                            container.tryBitmapRecycle()

                                            if (container.faceList!!.isNotEmpty()) {
                                                val disposable = api.uploadAnnotation(
                                                        container.fileName,
                                                        container.faceList!!
                                                )
                                                        .subscribeOn(Schedulers.newThread())
                                                        .observeOn(AndroidSchedulers.mainThread())
                                                        .subscribe({
                                                            textView.text = "%s (%,d/%,d)"
                                                                    .format(container.fileName,
                                                                            processCount,
                                                                            totalCount)
                                                            imageView.setImageBitmap(container.bitmap)
                                                        }, {})
                                                compositeDisposable.add(disposable)
                                            }
                                        }
                            })
                            .doOnComplete({
                                nextBatch()
                            })
                            .subscribe()
                    compositeDisposable.add(disposable)
                }
                .subscribe())
    }

    override fun onDestroy() {
        super.onDestroy()

        compositeDisposable.dispose()
        faceDetector.close()

        logFileOutputStream?.close()
        logFileOutputStream = null
    }

    class Container(val fileName: String,
                    val bitmap: Bitmap) {

        var resizedBitmap: Bitmap? = null
        var elapsedTime: Long? = null
        var faceList: List<Face>? = null

        fun tryBitmapRecycle() {
            resizedBitmap?.let {
                if (it != bitmap) {
                    it.recycle()
                }
            }
        }
    }

    class ConvertToFaceList(val container: Container, val visionFaceList: List<FirebaseVisionFace>)
        : Single<Container>() {
        override fun subscribeActual(observer: SingleObserver<in Container>) {

            container.faceList = visionFaceList.map {
                convertToJson(container.fileName, container.resizedBitmap!!, it)
            }
            observer.onSuccess(container)
        }


        private fun convertToJson(fileName: String,
                                  bitmap: Bitmap,
                                  visionFace: FirebaseVisionFace): Face {

            val face = Face(fileName, bitmap, visionFace.boundingBox)

            visionFace.getLandmark(FirebaseVisionFaceLandmark.LEFT_EYE)?.let {
                face.setLeftEye(it.position)
            }
            visionFace.getLandmark(FirebaseVisionFaceLandmark.RIGHT_EYE)?.let {
                face.setRightEye(it.position)
            }

            visionFace.getLandmark(FirebaseVisionFaceLandmark.NOSE_BASE)?.let {
                face.setNoseBase(it.position)
            }

            visionFace.getLandmark(FirebaseVisionFaceLandmark.LEFT_MOUTH)?.let {
                face.setLeftMouth(it.position)
            }
            visionFace.getLandmark(FirebaseVisionFaceLandmark.RIGHT_MOUTH)?.let {
                face.setRightMouth(it.position)
            }
            visionFace.getLandmark(FirebaseVisionFaceLandmark.BOTTOM_MOUTH)?.let {
                face.setBottomMouth(it.position)
            }

            visionFace.getLandmark(FirebaseVisionFaceLandmark.LEFT_CHEEK)?.let {
                face.setRightCheek(it.position)
            }
            visionFace.getLandmark(FirebaseVisionFaceLandmark.RIGHT_CHEEK)?.let {
                face.setLeftCheek(it.position)
            }

            visionFace.getLandmark(FirebaseVisionFaceLandmark.LEFT_EAR)?.let {
                face.setLeftEar(it.position)
            }
            visionFace.getLandmark(FirebaseVisionFaceLandmark.RIGHT_EAR)?.let {
                face.setRightEar(it.position)
            }

            return face
        }
    }

    class GetLastOffset(val inputStream: InputStream) : Single<Int>() {
        override fun subscribeActual(observer: SingleObserver<in Int>) {
            inputStream.use {
                val reader = InputStreamReader(it)
                val lines = reader.readLines()
                if (lines.isEmpty()) {
                    observer.onSuccess(0)
                    return
                }
                val lastLine = lines.last()
                val offset = lastLine.split(",")[0].toInt()

                Log.d(TAG, "offset: %d".format(offset))
                observer.onSuccess(offset)
            }
        }
    }

    class ResizeBitmap(var container: Container) : Single<Container>() {
        override fun subscribeActual(observer: SingleObserver<in Container>) {
            val bitmap = container.bitmap
            container.resizedBitmap = Bitmap.createScaledBitmap(bitmap,
                    256,
                    256,
                    true)

            observer.onSuccess(container)
        }
    }

    class CreateBitmap(val stream: InputStream) : Single<Bitmap>() {
        override fun subscribeActual(observer: SingleObserver<in Bitmap>) {
            val bitmap = BitmapFactory.decodeStream(stream)
            observer.onSuccess(bitmap)
        }
    }

    class LoadBitmapFromAsset(val assetManager: AssetManager, val fileName: String) : Single<Bitmap>() {
        override fun subscribeActual(observer: SingleObserver<in Bitmap>) {
            assetManager.open(fileName).use {
                val bitmap = BitmapFactory.decodeStream(it)
                observer.onSuccess(bitmap)
            }
        }
    }

    var logFileOutputStream: Writer? = null

    inner class DetectFace(val container: Container) : Single<Pair<Container, List<FirebaseVisionFace>>>() {
        override fun subscribeActual(observer: SingleObserver<in Pair<Container, List<FirebaseVisionFace>>>) {
            container.resizedBitmap?.let {

                val visionImage = FirebaseVisionImage.fromBitmap(it);

                val start = Debug.threadCpuTimeNanos()

                faceDetector.detectInImage(visionImage)
                        .addOnSuccessListener {
                            val elapsed = (Debug.threadCpuTimeNanos() - start) / 1000000
                            Log.d(TAG, "Elapsed %d - %,d".format(processCount, elapsed))

                            logFileOutputStream?.apply {
                                write("%d,%d\n".format(processCount, elapsed))
                                flush()
                            }

                            saveToElapsed(elapsed)
                            container.elapsedTime = elapsed
                            observer.onSuccess(Pair(container, it))
                        }
                        .addOnFailureListener(
                                object : OnFailureListener {
                                    override fun onFailure(@NonNull e: Exception) {
                                        observer.onError(e)
                                    }
                                })
            }
        }

        private fun saveToElapsed(elapsed: Long) {
        }
    }
}

