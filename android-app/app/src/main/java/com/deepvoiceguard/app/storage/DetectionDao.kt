package com.deepvoiceguard.app.storage

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface DetectionDao {

    @Insert
    suspend fun insert(detection: DetectionEntity): Long

    @Query("SELECT * FROM detections ORDER BY timestamp DESC")
    fun getAll(): Flow<List<DetectionEntity>>

    @Query("SELECT * FROM detections WHERE timestamp BETWEEN :startMs AND :endMs ORDER BY timestamp DESC")
    fun getByDateRange(startMs: Long, endMs: Long): Flow<List<DetectionEntity>>

    @Query("SELECT * FROM detections WHERE threatLevel IN ('WARNING', 'DANGER') ORDER BY timestamp DESC")
    fun getHighThreat(): Flow<List<DetectionEntity>>

    @Query("SELECT COUNT(*) FROM detections WHERE timestamp >= :sinceMs")
    suspend fun countSince(sinceMs: Long): Int

    @Query("SELECT * FROM detections WHERE timestamp < :beforeMs")
    suspend fun getOlderThan(beforeMs: Long): List<DetectionEntity>

    @Query("DELETE FROM detections WHERE timestamp < :beforeMs")
    suspend fun deleteOlderThan(beforeMs: Long): Int

    @Query("SELECT * FROM detections ORDER BY timestamp ASC LIMIT 1")
    suspend fun getOldest(): DetectionEntity?

    @Query("DELETE FROM detections WHERE id = :id")
    suspend fun deleteById(id: Long)

    // --- v3 피싱 탐지 쿼리 ---

    @Query("SELECT * FROM detections WHERE phishingScore > :threshold ORDER BY timestamp DESC")
    fun getPhishingDetections(threshold: Float = 0.1f): Flow<List<DetectionEntity>>

    @Query("SELECT * FROM detections WHERE combinedThreatLevel IN ('WARNING', 'DANGER', 'CRITICAL') ORDER BY timestamp DESC")
    fun getCombinedHighThreat(): Flow<List<DetectionEntity>>

    @Query("SELECT COUNT(*) FROM detections WHERE phishingScore > 0.1 AND timestamp >= :sinceMs")
    suspend fun countPhishingSince(sinceMs: Long): Int

    /** 참조된 audioFilePath 집합 (CleanupWorker unreferenced file sweep용). */
    @Query("SELECT audioFilePath FROM detections WHERE audioFilePath IS NOT NULL")
    suspend fun getAllAudioFilePaths(): List<String>
}
