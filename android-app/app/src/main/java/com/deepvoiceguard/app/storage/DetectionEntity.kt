package com.deepvoiceguard.app.storage

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "detections")
data class DetectionEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val timestamp: Long,            // System.currentTimeMillis()
    val fakeScore: Float,
    val realScore: Float,
    val confidence: Float,
    val durationMs: Long,           // 오디오 세그먼트 길이
    val audioFilePath: String?,     // 암호화된 오디오 파일 경로 (nullable)
    val inferenceMode: String,      // "on_device" or "server"
    val threatLevel: String,        // "SAFE", "CAUTION", "WARNING", "DANGER"
    val latencyMs: Long,
    val callSessionId: String? = null, // 통화 세션 ID (통화 중 감지 시)
)
