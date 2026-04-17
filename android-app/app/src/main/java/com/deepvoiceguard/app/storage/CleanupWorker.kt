package com.deepvoiceguard.app.storage

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
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
        private const val ONESHOT_WORK_PREFIX = "deepvoice_cleanup_oneshot_"
        private const val DEFAULT_RETENTION_DAYS = 7L
        private const val MAX_STORAGE_BYTES = 100L * 1024 * 1024  // 100MB
        const val EXTRA_ORPHAN_PATH = "orphan_path"

        fun schedule(context: Context) {
            val request = PeriodicWorkRequestBuilder<CleanupWorker>(1, TimeUnit.DAYS)
                .build()
            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                WORK_NAME,
                ExistingPeriodicWorkPolicy.KEEP,
                request,
            )
        }

        /**
         * 특정 orphan 파일을 즉시 정리하도록 one-shot 작업 스케줄.
         * HistoryScreen delete에서 file 삭제 실패 시 호출 — age와 무관하게 즉시 retry.
         */
        fun scheduleOrphanCleanup(context: Context, orphanPath: String) {
            val request = OneTimeWorkRequestBuilder<CleanupWorker>()
                .setInputData(
                    Data.Builder().putString(EXTRA_ORPHAN_PATH, orphanPath).build()
                )
                .build()
            WorkManager.getInstance(context).enqueueUniqueWork(
                ONESHOT_WORK_PREFIX + orphanPath.hashCode(),
                ExistingWorkPolicy.KEEP,
                request,
            )
        }
    }

    override suspend fun doWork(): Result {
        return try {
            val cutoffMs = System.currentTimeMillis() - TimeUnit.DAYS.toMillis(DEFAULT_RETENTION_DAYS)
            val db = DetectionDatabase.getInstance(applicationContext)
            val dao = db.detectionDao()
            val storage = EncryptedStorage(applicationContext)

            // 0. **One-shot orphan cleanup** — HistoryScreen delete에서 file 삭제 실패한 경우.
            // age 무관하게 즉시 정리 시도.
            inputData.getString(EXTRA_ORPHAN_PATH)?.let { orphanPath ->
                val deleted = storage.deleteSegment(orphanPath)
                Log.d(TAG, "Orphan one-shot cleanup: $orphanPath -> success=$deleted")
                if (deleted) return Result.success()
                Log.w(TAG, "Orphan one-shot cleanup failed: $orphanPath")
                // 실패 시 retry 가능하도록 Result.retry() 반환.
                return Result.retry()
            }

            // 1. DB에서 오래된 레코드 조회 → 연결된 파일 삭제 → DB 레코드 삭제
            val oldRecords = dao.getOlderThan(cutoffMs)
            for (record in oldRecords) {
                record.audioFilePath?.let { path ->
                    storage.deleteSegment(path)
                }
            }
            val deletedRecords = dao.deleteOlderThan(cutoffMs)
            Log.d(TAG, "Deleted $deletedRecords detection rows (file deletion logged separately)")

            // 2. **DB-aware 일원화된 파일 정리 (H86)**.
            val cleanupResult = deleteUnreferencedFiles(storage, dao)
            if (cleanupResult.deleted > 0) {
                Log.d(TAG, "Deleted ${cleanupResult.deleted} unreferenced encrypted files")
            }
            if (cleanupResult.failed > 0) {
                // **H88**: 삭제 실패한 파일이 남아 있으면 retry. quota가 valid history를 지우지 않게.
                Log.w(TAG, "deleteUnreferencedFiles failed on ${cleanupResult.failed} files; retrying later")
                return Result.retry()
            }
            if (cleanupResult.remaining > 0) {
                // **H89**: age-gate로 skip된 unreferenced(fresh .pending/.enc)가 여전히 디스크
                // 공간 점유 → quota loop가 valid row를 지워서 orphan을 상쇄하는 것 방지.
                Log.w(
                    TAG,
                    "deleteUnreferencedFiles: ${cleanupResult.remaining} fresh unreferenced files " +
                        "still present; skipping quota enforcement to preserve valid history"
                )
                return Result.success()
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

            Result.success()
        } catch (t: Throwable) {
            Log.w(TAG, "cleanup failed", t)
            Result.retry()
        }
    }

    /**
     * DB에 참조되지 않는 encrypted 파일을 삭제한다 (age 무관).
     * HistoryScreen delete에서 file 삭제 실패 후 CleanupWorker가 다음 주기에 자동 정리.
     */
    data class UnreferencedCleanupResult(
        val deleted: Int,
        val failed: Int,
        /** age-gate 등으로 skip되어 여전히 남은 unreferenced 파일 수. quota skip용. */
        val remaining: Int,
    )

    /**
     * @return UnreferencedCleanupResult — deleted/failed/remaining.
     *         failed + remaining > 0이면 quota loop 건너뛰는 것이 안전.
     */
    private suspend fun deleteUnreferencedFiles(
        storage: EncryptedStorage,
        dao: DetectionDao,
    ): UnreferencedCleanupResult {
        val allPaths = storage.listAllPaths()
        if (allPaths.isEmpty()) return UnreferencedCleanupResult(0, 0, 0)
        val referenced = dao.getAllAudioFilePaths().toSet()
        val now = System.currentTimeMillis()
        // **.pending = 영구 확장자 (H84)**: DB 참조 있으면 절대 삭제 금지(이미 위에서 continue).
        // DB 참조 없는 unreferenced .pending은 10분 age 후 abandoned로 정리.
        val stalePendingCutoff = now - 10 * 60_000L
        // **.enc = legacy 확장자**: 예전 rename 시스템의 잔존 파일. 60초 이내면 혹시 모를
        // write 진행 중 보호, 이상이면 unreferenced orphan으로 정리.
        val staleEncCutoff = now - 60_000L
        var deleted = 0
        var failed = 0
        var remaining = 0
        for (path in allPaths) {
            if (path in referenced) continue  // DB에 있으면 절대 삭제 안 함.
            val file = File(path)
            val cutoff = when {
                path.endsWith(".pending") -> stalePendingCutoff
                path.endsWith(".enc") -> staleEncCutoff
                else -> now
            }
            if (file.lastModified() > cutoff) {
                // age-skipped unreferenced — quota enforcement에서 이것이 있으면 valid row
                // 지우는 것을 방지해야 하므로 카운팅.
                remaining++
                continue
            }
            if (storage.deleteSegment(path)) deleted++ else failed++
        }
        return UnreferencedCleanupResult(deleted, failed, remaining)
    }
}
