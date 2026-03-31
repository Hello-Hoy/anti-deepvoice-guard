package com.deepvoiceguard.app.storage

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * 주기적으로 오래된 탐지 기록과 암호화 오디오 파일을 삭제한다.
 * DB 레코드의 audioFilePath를 사용하여 DB/파일 일관성을 유지한다.
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
        val db = DetectionDatabase.getInstance(applicationContext)
        val dao = db.detectionDao()
        val storage = EncryptedStorage(applicationContext)

        // 1. DB에서 오래된 레코드 조회 → 연결된 파일 삭제 → DB 레코드 삭제
        val oldRecords = dao.getOlderThan(cutoffMs)
        for (record in oldRecords) {
            record.audioFilePath?.let { path ->
                storage.deleteSegment(path)
            }
        }
        val deletedRecords = dao.deleteOlderThan(cutoffMs)
        Log.d(TAG, "Deleted $deletedRecords old detection records and their audio files")

        // 2. 고아 파일 정리 (DB에 없는 파일)
        val orphanDeleted = storage.deleteOlderThan(cutoffMs)
        if (orphanDeleted > 0) {
            Log.d(TAG, "Deleted $orphanDeleted orphaned encrypted files")
        }

        // 3. 저장 용량 초과 시 가장 오래된 것부터 삭제 (반복)
        var iterations = 0
        while (storage.totalStorageBytes() > MAX_STORAGE_BYTES && iterations < 10) {
            val oldest = dao.getOldest()
            if (oldest == null) break
            oldest.audioFilePath?.let { storage.deleteSegment(it) }
            dao.deleteById(oldest.id)
            iterations++
        }
        if (iterations > 0) {
            Log.d(TAG, "Storage cap enforcement: deleted $iterations records to stay under ${MAX_STORAGE_BYTES / 1024 / 1024}MB")
        }

        return Result.success()
    }
}
