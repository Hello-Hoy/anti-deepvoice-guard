package com.deepvoiceguard.app.storage

import android.content.Context
import android.util.Log
import androidx.security.crypto.EncryptedFile
import androidx.security.crypto.MasterKey
import java.io.File
import java.io.IOException
import java.security.GeneralSecurityException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * AES-256-GCM으로 의심 오디오 세그먼트를 암호화 저장한다.
 * AndroidKeyStore에서 마스터키를 관리한다.
 */
class EncryptedStorage(private val context: Context) {

    companion object {
        private const val TAG = "EncryptedStorage"
    }

    private val masterKey = MasterKey.Builder(context)
        .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
        .build()

    private val storageDir: File
        get() {
            val dir = File(context.filesDir, "encrypted_segments")
            if (!dir.exists()) dir.mkdirs()
            return dir
        }

    /**
     * 오디오 세그먼트를 `.pending` 확장자로 암호화 저장한다.
     *
     * **.pending은 영구 확장자**: DB row가 이 path 그대로 참조. CleanupWorker는 DB에서
     * 참조된 `.pending`은 절대 삭제하지 않고, DB 참조 없는 orphan만 10분 age 후 삭제한다.
     * rename을 하지 않아 commit-before-insert race가 원천 제거됨 (H84).
     */
    fun saveSegment(audio: FloatArray): String {
        val timestamp = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss-SSS", Locale.US).format(Date())
        val file = File(storageDir, "$timestamp.pending")

        val encryptedFile = EncryptedFile.Builder(
            context,
            file,
            masterKey,
            EncryptedFile.FileEncryptionScheme.AES256_GCM_HKDF_4KB,
        ).build()

        encryptedFile.openFileOutput().use { output ->
            // 헤더: sample count (4 bytes)
            val sampleCount = audio.size
            output.write(sampleCount shr 24 and 0xFF)
            output.write(sampleCount shr 16 and 0xFF)
            output.write(sampleCount shr 8 and 0xFF)
            output.write(sampleCount and 0xFF)

            // PCM 16-bit로 변환하여 저장 (50% 크기 절약)
            for (sample in audio) {
                val clamped = sample.coerceIn(-1f, 1f)
                val intVal = (clamped * 32767).toInt().toShort()
                output.write(intVal.toInt() and 0xFF)
                output.write(intVal.toInt() shr 8 and 0xFF)
            }
        }

        return file.absolutePath
    }

    /**
     * 암호화된 오디오 파일을 복호화하여 Float 배열로 반환한다.
     */
    fun loadSegment(filePath: String): FloatArray? {
        val file = File(filePath)
        return try {
            if (!file.exists() || !isUnderStorageDir(file)) return null

            val encryptedFile = EncryptedFile.Builder(
                context,
                file,
                masterKey,
                EncryptedFile.FileEncryptionScheme.AES256_GCM_HKDF_4KB,
            ).build()

            encryptedFile.openFileInput().use { input ->
                val bytes = input.readBytes()
                if (bytes.size < 4) return@use null

                val sampleCount = (bytes[0].toInt() and 0xFF shl 24) or
                        (bytes[1].toInt() and 0xFF shl 16) or
                        (bytes[2].toInt() and 0xFF shl 8) or
                        (bytes[3].toInt() and 0xFF)

                // sampleCount 검증: 음수 또는 비정상적으로 큰 값 방어
                val maxSamples = 16000 * 30  // 최대 30초
                if (sampleCount <= 0 || sampleCount > maxSamples) return@use null

                val expectedBytes = 4 + sampleCount * 2
                if (bytes.size < expectedBytes) return@use null

                val audio = FloatArray(sampleCount)
                for (i in 0 until sampleCount) {
                    val offset = 4 + i * 2
                    val lo = bytes[offset].toInt() and 0xFF
                    val hi = bytes[offset + 1].toInt() and 0xFF
                    val shortVal = (hi shl 8 or lo).toShort()
                    audio[i] = shortVal / 32768f
                }
                audio
            }
        } catch (t: GeneralSecurityException) {
            Log.w(TAG, "loadSegment failed: $filePath", t)
            null
        } catch (t: IOException) {
            Log.w(TAG, "loadSegment failed: $filePath", t)
            null
        } catch (t: Throwable) {
            Log.w(TAG, "loadSegment failed: $filePath", t)
            null  // 복호화/키스토어 실패 시 fail-closed
        }
    }

    /**
     * 특정 파일 삭제 (storageDir 하위만 허용).
     * **ENOENT는 success로 간주** — 이미 없는 파일은 삭제 의도가 충족된 상태.
     * HistoryScreen atomic delete flow에서 undeletable row를 만들지 않기 위함.
     */
    fun deleteSegment(filePath: String): Boolean {
        val file = File(filePath)
        if (!isUnderStorageDir(file)) return false
        if (!file.exists()) return true  // ENOENT → already gone, success.
        return file.delete()
    }

    private fun isUnderStorageDir(file: File): Boolean {
        return file.canonicalPath.startsWith(storageDir.canonicalPath)
    }

    /** 저장된 파일 총 크기 (bytes). */
    fun totalStorageBytes(): Long {
        return storageDir.listFiles()?.sumOf { it.length() } ?: 0
    }

    /** storageDir 내 전체 파일 경로 목록 (orphan sweep용). */
    fun listAllPaths(): List<String> =
        storageDir.listFiles()?.map { it.absolutePath } ?: emptyList()
}
