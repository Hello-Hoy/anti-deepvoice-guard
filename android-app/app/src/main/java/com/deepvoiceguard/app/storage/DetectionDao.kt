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

    @Query("DELETE FROM detections WHERE timestamp < :beforeMs")
    suspend fun deleteOlderThan(beforeMs: Long): Int

    @Query("DELETE FROM detections WHERE id = :id")
    suspend fun deleteById(id: Long)
}
