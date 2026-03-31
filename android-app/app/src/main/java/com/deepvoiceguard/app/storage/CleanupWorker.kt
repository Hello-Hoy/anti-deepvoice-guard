package com.deepvoiceguard.app.storage

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import java.util.concurrent.TimeUnit

/**
 * 주기적으로 오래된 탐지 기록과 암호화 오디오 파일을 삭제한다.
 */
class CleanupWorker(
    context: Context,
    params: WorkerParameters,
) : CoroutineWorker(context, params) {

    companion object {
        private const val TAG = "CleanupWorker"
        private const val WORK_NAME = "deepvoice_cleanup"
        private const val DEFAULT_RETENTION_DAYS = 7L
        private const val MAX_STORAGE_BYTES = 100L * 1024 * 1024  // 100MB

        fun schedule(context: Context) {
            val request = PeriodicWorkRequestBuilder<CleanupWorker>(1, TimeUnit.DAYS)
                .build()
            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                WORK_NAME,
                ExistingPeriodicWorkPolicy.KEEP,
                request,
            )
        }
    }

    override suspend fun doWork(): Result {
        val cutoffMs = System.currentTimeMillis() - TimeUnit.DAYS.toMillis(DEFAULT_RETENTION_DAYS)

        // DB 레코드 삭제
        val db = DetectionDatabase.getInstance(applicationContext)
        val deletedRecords = db.detectionDao().deleteOlderThan(cutoffMs)
        Log.d(TAG, "Deleted $deletedRecords old detection records")

        // 암호화 파일 삭제
        val storage = EncryptedStorage(applicationContext)
        val deletedFiles = storage.deleteOlderThan(cutoffMs)
        Log.d(TAG, "Deleted $deletedFiles old encrypted files")

        // 저장 용량 초과 시 추가 삭제
        if (storage.totalStorageBytes() > MAX_STORAGE_BYTES) {
            val aggressiveCutoff = System.currentTimeMillis() - TimeUnit.DAYS.toMillis(1)
            storage.deleteOlderThan(aggressiveCutoff)
            db.detectionDao().deleteOlderThan(aggressiveCutoff)
            Log.d(TAG, "Storage exceeded ${MAX_STORAGE_BYTES / 1024 / 1024}MB, aggressive cleanup done")
        }

        return Result.success()
    }
}
