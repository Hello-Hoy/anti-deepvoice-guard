package com.deepvoiceguard.app.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.deepvoiceguard.app.storage.CleanupWorker
import com.deepvoiceguard.app.storage.DetectionDatabase
import com.deepvoiceguard.app.storage.DetectionEntity
import com.deepvoiceguard.app.storage.EncryptedStorage
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun HistoryScreen() {
    val context = LocalContext.current
    val db = remember { DetectionDatabase.getInstance(context) }
    val encryptedStorage = remember { EncryptedStorage(context) }
    val detections by db.detectionDao().getAll().collectAsState(initial = emptyList())

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
    ) {
        Text(
            "Detection History",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
        )

        Spacer(modifier = Modifier.height(8.dp))
        Text("${detections.size} records", style = MaterialTheme.typography.bodySmall)
        Spacer(modifier = Modifier.height(16.dp))

        if (detections.isEmpty()) {
            Text(
                "No detections yet.\nStart monitoring to see results here.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.outline,
            )
        } else {
            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(detections, key = { it.id }) { detection ->
                    DetectionCard(detection) {
                        CoroutineScope(Dispatchers.IO).launch {
                            // **DB-first 순서** — DB row가 canonical. row 삭제 성공 후 file 삭제 시도.
                            //  - DB 실패: 아무 상태 변경 없음 (file 유지, row 유지) — 재시도 가능.
                            //  - DB 성공, file 실패: row는 사라지고 file이 orphan. CleanupWorker가
                            //    age/quota 기준으로 나중에 자동 삭제. data-loss는 의도된 delete.
                            val path = detection.audioFilePath
                            val dbOk = runCatching { db.detectionDao().deleteById(detection.id) }
                                .isSuccess
                            if (!dbOk) {
                                android.util.Log.w("HistoryScreen", "history delete — DB row removal failed")
                                return@launch
                            }
                            if (path != null) {
                                val fileOk = runCatching { encryptedStorage.deleteSegment(path) }
                                    .getOrDefault(false)
                                if (!fileOk) {
                                    android.util.Log.w(
                                        "HistoryScreen",
                                        "history delete — DB row removed but file cleanup failed: $path. " +
                                            "Scheduling one-shot CleanupWorker retry."
                                    )
                                    // **즉시 one-shot cleanup 스케줄** — age와 무관하게 retry.
                                    CleanupWorker.scheduleOrphanCleanup(context, path)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun DetectionCard(detection: DetectionEntity, onDelete: () -> Unit) {
    val dateFormat = remember { SimpleDateFormat("HH:mm:ss", Locale.getDefault()) }
    // **combinedThreatLevel이 canonical severity** — CRITICAL 등 escalation 표시.
    // 레거시 레코드(combinedThreatLevel="SAFE" 기본값)도 threatLevel로 fallback.
    val effectiveLevel = detection.combinedThreatLevel.takeUnless { it == "SAFE" }
        ?: detection.threatLevel
    val containerColor = when (effectiveLevel) {
        "CRITICAL" -> MaterialTheme.colorScheme.errorContainer
        "DANGER" -> MaterialTheme.colorScheme.errorContainer
        "WARNING" -> MaterialTheme.colorScheme.tertiaryContainer
        "CAUTION" -> MaterialTheme.colorScheme.secondaryContainer
        else -> MaterialTheme.colorScheme.surfaceVariant
    }
    val icon = when (effectiveLevel) {
        "CRITICAL" -> "🔥"
        "DANGER" -> "🚨"
        "WARNING" -> "⚠️"
        "CAUTION" -> "📋"
        else -> "✅"
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = containerColor),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    "$icon ${dateFormat.format(Date(detection.timestamp))} | " +
                            "$effectiveLevel | " +
                            "Score: ${String.format("%.2f", detection.fakeScore)}",
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    "${detection.durationMs / 1000f}s | " +
                            "${detection.inferenceMode} | " +
                            "${detection.latencyMs}ms" +
                            (if (detection.phishingScore > 0.1f) " | 피싱 ${String.format("%.2f", detection.phishingScore)}" else ""),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            IconButton(onClick = onDelete) {
                Icon(Icons.Default.Delete, contentDescription = "Delete")
            }
        }
    }
}
