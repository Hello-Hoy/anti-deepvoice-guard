# ONNX Runtime — keep JNI classes
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# Room
-keep class * extends androidx.room.RoomDatabase
-keep @androidx.room.Entity class *
-dontwarn androidx.room.paging.**

# Retrofit + Gson
-keepattributes Signature
-keepattributes *Annotation*
-keep class com.deepvoiceguard.app.inference.ServerDetectionResponse { *; }
-dontwarn retrofit2.**
-dontwarn okhttp3.**
