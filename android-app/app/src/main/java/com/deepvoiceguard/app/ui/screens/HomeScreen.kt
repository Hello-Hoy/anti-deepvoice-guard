package com.deepvoiceguard.app.ui.screens

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.deepvoiceguard.app.inference.ThreatLevel
import com.deepvoiceguard.app.service.AudioCaptureService
import com.deepvoiceguard.app.service.AudioPipeline

@Composable
fun HomeScreen() {
    val context = LocalContext.current
    var isMonitoring by remember { mutableStateOf(false) }
    var pipeline by remember { mutableStateOf<AudioPipeline?>(null) }

    val latestResult by pipeline?.latestResult?.collectAsState() ?: remember { mutableStateOf(null) }
    val vadProb by pipeline?.vadProbability?.collectAsState() ?: remember { mutableStateOf(0f) }

    // Service 연결
    val connection = remember {
        object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
                val binder = service as? AudioCaptureService.LocalBinder
                pipeline = binder?.pipeline
            }
            override fun onServiceDisconnected(name: ComponentName?) {
                pipeline = null
            }
        }
    }

    DisposableEffect(Unit) {
        val intent = Intent(context, AudioCaptureService::class.java)
        context.bindService(intent, connection, Context.BIND_AUTO_CREATE)
        onDispose { context.unbindService(connection) }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = "Anti-DeepVoice Guard",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
        )

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = if (isMonitoring) "Monitoring Active" else "Monitoring Stopped",
            color = if (isMonitoring) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.outline,
        )

        Spacer(modifier = Modifier.height(32.dp))

        // Start/Stop 버튼
        Button(
            onClick = {
                val intent = Intent(context, AudioCaptureService::class.java)
                if (!isMonitoring) {
                    intent.action = AudioCaptureService.ACTION_START
                    context.startForegroundService(intent)
                } else {
                    intent.action = AudioCaptureService.ACTION_STOP
                    context.startService(intent)
                }
                isMonitoring = !isMonitoring
            },
            modifier = Modifier.size(120.dp),
            shape = CircleShape,
            colors = ButtonDefaults.buttonColors(
                containerColor = if (isMonitoring) MaterialTheme.colorScheme.error
                                 else MaterialTheme.colorScheme.primary,
            ),
        ) {
            Text(
                text = if (isMonitoring) "STOP" else "START",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
            )
        }

        Spacer(modifier = Modifier.height(32.dp))

        // VAD 실시간 표시
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant,
            ),
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Voice Activity", style = MaterialTheme.typography.titleSmall)
                Spacer(modifier = Modifier.height(8.dp))
                LinearProgressIndicator(
                    progress = { vadProb },
                    modifier = Modifier.fillMaxWidth(),
                )
                Text("${(vadProb * 100).toInt()}%", style = MaterialTheme.typography.bodySmall)
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // 탐지 결과
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = when (latestResult?.threatLevel) {
                    ThreatLevel.DANGER -> MaterialTheme.colorScheme.errorContainer
                    ThreatLevel.WARNING -> MaterialTheme.colorScheme.tertiaryContainer
                    ThreatLevel.CAUTION -> MaterialTheme.colorScheme.secondaryContainer
                    else -> MaterialTheme.colorScheme.surfaceVariant
                },
            ),
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Detection Status", style = MaterialTheme.typography.titleSmall)
                Spacer(modifier = Modifier.height(8.dp))

                val threatText = when (latestResult?.threatLevel) {
                    ThreatLevel.DANGER -> "DANGER - Deepfake Detected!"
                    ThreatLevel.WARNING -> "WARNING - Suspicious Voice"
                    ThreatLevel.CAUTION -> "CAUTION - Elevated Risk"
                    ThreatLevel.SAFE -> "SAFE"
                    null -> "No data yet"
                }
                Text(threatText, fontWeight = FontWeight.Bold)

                latestResult?.let { result ->
                    Spacer(modifier = Modifier.height(4.dp))
                    Text("Fake Score: ${String.format("%.1f%%", result.averageFakeScore * 100)}")
                    Text("Latency: ${result.latestResult.latencyMs}ms")
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // 오늘 통계
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Today's Stats", style = MaterialTheme.typography.titleSmall)
                Spacer(modifier = Modifier.height(8.dp))
                val stats = pipeline?.getStats()
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Column {
                        Text("Segments", style = MaterialTheme.typography.bodySmall)
                        Text("${stats?.segmentsAnalyzed ?: 0}", fontWeight = FontWeight.Bold)
                    }
                    Column {
                        Text("Detections", style = MaterialTheme.typography.bodySmall)
                        Text("${stats?.detectionsCount ?: 0}", fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }
}
