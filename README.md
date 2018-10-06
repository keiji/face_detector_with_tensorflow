# FaceDetector with TensorFlow

## Description
2018年10月8日池袋サンシャインで開催される技術系同人誌オンリーイベント「技術書典5」で頒布予定のTensorFlow本「Liteな期待が重すぎる！」のサンプルプロジェクトです。

![表紙画像](https://blog.keiji.io/wp-content/uploads/2018/10/Lite_B5_hyoushi_4mm_1002_86-KB.jpg)

|タイトル|Liteな期待が重すぎる！ （茶色いトイプーは食べ物じゃないっ!?）|
|:---|:---|
|判型|B5 PDF|
|ページ数|本文60p|
|頒布価格|1,000円|
|発行|めがねをかけるんだ|
|配置|え33|

## Install app
You can download the APK file from [release page](https://github.com/keiji/face_detector_with_tensorflow/releases/latest).

## Install model file
The TensorFlow model file(`food_model_4ch.pb`) is not packaged in the repo because of their size.
You can download the from [release page](https://github.com/keiji/face_detector_with_tensorflow/releases/latest) to the assets directory in the source tree.

```
Copyright 2017 The Android Open Source Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

```
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
```

----
モデルの学習には、[さくらインターネット株式会社](https://www.sakura.ad.jp/) の [高火力 専用GPUサーバー](https://www.sakura.ad.jp/koukaryoku/) を利用しています。

The model is trained on [KOUKARYOKU](https://www.sakura.ad.jp/koukaryoku/) dedicated GPU server hosting by [Sakura Internet Inc](https://www.sakura.ad.jp/).
