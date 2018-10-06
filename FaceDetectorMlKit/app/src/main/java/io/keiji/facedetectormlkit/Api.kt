package io.keiji.facedetectormlkit

import io.reactivex.Single
import okhttp3.ResponseBody
import retrofit2.http.*

interface Api {

    @GET("/")
    fun getFileList(
            @Query("limit") limit: Int?,
            @Query("offset") offset: Int?
    ): Single<List<String>>

    @GET("/{file_name}")
    fun getBitmap(
            @Path("file_name") fileName: String
    ): Single<ResponseBody>

    @PUT("/{file_name}")
    fun uploadAnnotation(@Path("file_name") fileName: String,
                         @Body resultList: List<Face>): Single<Unit>

}