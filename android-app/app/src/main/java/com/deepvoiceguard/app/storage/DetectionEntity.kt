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
    val callSessionId: String? = null,       // 통화 세션 ID (통화 중 감지 시)
    // --- v3 피싱 탐지 필드 (nullable/default — 마이그레이션 안전) ---
    val phishingScore: Float = 0f,           // 피싱 점수 0.0~1.0
    val phishingKeywords: String? = null,    // 콤마 구분 "검찰,계좌이체,긴급"
    val transcription: String? = null,       // 전사 텍스트 (opt-in 동의 시에만 저장)
    val combinedThreatLevel: String = "SAFE", // "SAFE"~"CRITICAL"
    val sttAvailable: Boolean = true,        // STT 가용 여부 기록
)
