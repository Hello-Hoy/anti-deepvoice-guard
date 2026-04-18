package com.deepvoiceguard.app.ui.screens

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.IBinder
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
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
import androidx.core.content.ContextCompat
import com.deepvoiceguard.app.inference.CombinedThreatLevel
import com.deepvoiceguard.app.inference.ThreatLevel
import com.deepvoiceguard.app.service.AudioCaptureService
import com.deepvoiceguard.app.service.CallDirection
import com.deepvoiceguard.app.service.CallSession
import com.deepvoiceguard.app.stt.SttStatus

@Composable
fun HomeScreen() {
    val context = LocalContext.current
    var service by remember { mutableStateOf<AudioCaptureService?>(null) }

    // 서비스의 live StateFlow를 구독 — combinedResult가 monitoring surface의 canonical source.
    val isMonitoring by service?.isMonitoring?.collectAsState() ?: remember { mutableStateOf(false) }
    val latestResult by service?.latestResult?.collectAsState() ?: remember { mutableStateOf(null) }
    val combinedResult by service?.combinedResult?.collectAsState()
        ?: remember { mutableStateOf(null) }
    val transcription by service?.transcription?.collectAsState()
        ?: remember { mutableStateOf("") }
    val vadProb by service?.vadProbability?.collectAsState() ?: remember { mutableStateOf(0f) }
    val stats by service?.stats?.collectAsState() ?: remember { mutableStateOf(null) }
    val callSession by service?.callSession?.collectAsState() ?: remember { mutableStateOf<CallSession?>(null) }
    var permissionDenied by remember { mutableStateOf(false) }

    // 모니터링 시작 helper
    fun startMonitoring() {
        val intent = Intent(context, AudioCaptureService::class.java).apply {
            action = AudioCaptureService.ACTION_START
        }
        context.startForegroundService(intent)
    }

    // 런타임 권한 요청 launcher
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { permissions ->
        if (permissions[Manifest.permission.RECORD_AUDIO] == true) {
            permissionDenied = false
            startMonitoring()
        } else {
            permissionDenied = true
        }
    }

    // 전화 상태 권한 launcher (통화 자동 감지용)
    val phonePermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { /* granted 여부와 무관하게 기능은 optional */ }

    // Service 연결
    val connection = remember {
        object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
                service = (binder as? AudioCaptureService.LocalBinder)?.service
            }
            override fun onServiceDisconnected(name: ComponentName?) {
                service = null
            }
        }
    }

    DisposableEffect(Unit) {
        val intent = Intent(context, AudioCaptureService::class.java)
        val bound = context.bindService(intent, connection, Context.BIND_AUTO_CREATE)
        onDispose { if (bound) runCatching { context.unbindService(connection) } }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
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

        // 통화 모니터링 배너
        if (callSession != null) {
            Spacer(modifier = Modifier.height(8.dp))
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                ),
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    val dirText = when (callSession?.direction) {
                        CallDirection.INCOMING -> "수신 통화"
                        CallDirection.OUTGOING -> "발신 통화"
                        else -> "통화"
                    }
                    Text(
                        "$dirText 모니터링 중",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                    )
                    callSession?.phoneNumber?.let { number ->
                        Text(number, style = MaterialTheme.typography.bodySmall)
                    }
                    Text(
                        "스피커폰 모드에서 상대방 음성 분석 가능",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }

        // STT 상태 배너 — combinedResult.sttStatus가 있으면 표시.
        combinedResult?.let { cr ->
            Spacer(modifier = Modifier.height(8.dp))
            val (sttText, sttColor) = when (cr.sttStatus) {
                SttStatus.LISTENING -> "STT 활성 (오프라인 처리)" to MaterialTheme.colorScheme.primaryContainer
                SttStatus.READY, SttStatus.PAUSED -> "STT 대기 중" to MaterialTheme.colorScheme.surfaceVariant
                SttStatus.CONSENT_NEEDED -> "STT 동의 필요 — 설정에서 활성화" to MaterialTheme.colorScheme.secondaryContainer
                SttStatus.UNAVAILABLE -> "STT 지원되지 않음 (딥보이스 탐지만 동작)" to MaterialTheme.colorScheme.surfaceVariant
                SttStatus.ERROR -> "STT 오류 — 오프라인 모델 확인 필요" to MaterialTheme.colorScheme.errorContainer
            }
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = sttColor),
            ) {
                Text(
                    sttText,
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(12.dp),
                )
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        // 권한 거부 안내
        if (permissionDenied) {
            Text(
                "마이크 권한이 필요합니다. 설정에서 권한을 허용해주세요.",
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall,
            )
            Spacer(modifier = Modifier.height(8.dp))
        }

        // Start/Stop 버튼
        Button(
            onClick = {
                if (!isMonitoring) {
                    // 런타임 권한 확인
                    val hasRecordAudio = ContextCompat.checkSelfPermission(
                        context, Manifest.permission.RECORD_AUDIO,
                    ) == PackageManager.PERMISSION_GRANTED

                    if (hasRecordAudio) {
                        permissionDenied = false
                        startMonitoring()
                    } else {
                        val permissions = buildList {
                            add(Manifest.permission.RECORD_AUDIO)
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                                add(Manifest.permission.POST_NOTIFICATIONS)
                            }
                        }
                        permissionLauncher.launch(permissions.toTypedArray())
                        // 전화 상태 권한도 별도 요청 (optional)
                        if (ContextCompat.checkSelfPermission(
                                context, Manifest.permission.READ_PHONE_STATE,
                            ) != PackageManager.PERMISSION_GRANTED
                        ) {
                            phonePermissionLauncher.launch(Manifest.permission.READ_PHONE_STATE)
                        }
                    }
                } else {
                    val intent = Intent(context, AudioCaptureService::class.java).apply {
                        action = AudioCaptureService.ACTION_STOP
                    }
                    context.startService(intent)
                }
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

        // **통합 위협 레벨 카드 — canonical monitoring surface**.
        // combinedResult가 활성이면 이것이 주 상태. 없으면 딥보이스 단독 표시로 fallback.
        val primaryLevel = combinedResult?.combinedThreatLevel
        val fakeScore = combinedResult?.deepfakeResult?.averageFakeScore
            ?: latestResult?.averageFakeScore ?: 0f
        val phishingScore = combinedResult?.phishingScore ?: 0f
        val matchedKeywords = combinedResult?.matchedKeywords.orEmpty()

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = primaryLevel?.let { combinedLevelContainer(it) }
                    ?: deepfakeLevelContainer(latestResult?.threatLevel)
            ),
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Threat Status", style = MaterialTheme.typography.titleSmall)
                Spacer(modifier = Modifier.height(8.dp))

                val (headline, detail) = when (primaryLevel) {
                    CombinedThreatLevel.CRITICAL ->
                        "CRITICAL — 딥보이스 + 보이스피싱 동시 감지!" to
                            "통합 위협이 최고 수준입니다. 즉시 통화 종료 권장."
                    CombinedThreatLevel.DANGER ->
                        "DANGER — 딥보이스 감지" to
                            "AI 합성 음성이 감지되었습니다."
                    CombinedThreatLevel.WARNING ->
                        "WARNING — 의심 징후" to
                            "보이스피싱/딥보이스 패턴이 감지되었습니다. 주의하세요."
                    CombinedThreatLevel.CAUTION ->
                        "CAUTION — 주의 필요" to
                            "경계 수준 징후가 있습니다."
                    CombinedThreatLevel.SAFE ->
                        "SAFE" to "현재 위험 징후 없음."
                    null -> when (latestResult?.threatLevel) {
                        ThreatLevel.DANGER -> "DANGER — Deepfake Detected!" to null
                        ThreatLevel.WARNING -> "WARNING — Suspicious Voice" to null
                        ThreatLevel.CAUTION -> "CAUTION — Elevated Risk" to null
                        ThreatLevel.SAFE -> "SAFE" to null
                        else -> "No data yet" to null
                    }
                }
                Text(headline, fontWeight = FontWeight.Bold)
                detail?.let {
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(it, style = MaterialTheme.typography.bodySmall)
                }

                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Column {
                        Text("Deepfake", style = MaterialTheme.typography.bodySmall)
                        Text(
                            String.format("%.1f%%", fakeScore * 100),
                            fontWeight = FontWeight.Bold,
                        )
                    }
                    Column {
                        Text("Phishing", style = MaterialTheme.typography.bodySmall)
                        Text(
                            String.format("%.1f%%", phishingScore * 100),
                            fontWeight = FontWeight.Bold,
                        )
                    }
                    Column {
                        Text("Latency", style = MaterialTheme.typography.bodySmall)
                        Text(
                            "${latestResult?.latestResult?.latencyMs ?: 0}ms",
                            fontWeight = FontWeight.Bold,
                        )
                    }
                }

                if (matchedKeywords.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        "Keywords: " + matchedKeywords.take(5).joinToString(", ") { it.keyword },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                }
            }
        }

        // 실시간 전사 카드 — STT 활성 시만.
        if (transcription.isNotBlank()) {
            Spacer(modifier = Modifier.height(16.dp))
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant,
                ),
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Live Transcription", style = MaterialTheme.typography.titleSmall)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = transcription.takeLast(500),
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // 오늘 통계
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Today's Stats", style = MaterialTheme.typography.titleSmall)
                Spacer(modifier = Modifier.height(8.dp))
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

@Composable
private fun combinedLevelContainer(level: CombinedThreatLevel) = when (level) {
    CombinedThreatLevel.CRITICAL, CombinedThreatLevel.DANGER ->
        MaterialTheme.colorScheme.errorContainer
    CombinedThreatLevel.WARNING -> MaterialTheme.colorScheme.tertiaryContainer
    CombinedThreatLevel.CAUTION -> MaterialTheme.colorScheme.secondaryContainer
    CombinedThreatLevel.SAFE -> MaterialTheme.colorScheme.surfaceVariant
}

@Composable
private fun deepfakeLevelContainer(level: ThreatLevel?) = when (level) {
    ThreatLevel.DANGER -> MaterialTheme.colorScheme.errorContainer
    ThreatLevel.WARNING -> MaterialTheme.colorScheme.tertiaryContainer
    ThreatLevel.CAUTION -> MaterialTheme.colorScheme.secondaryContainer
    else -> MaterialTheme.colorScheme.surfaceVariant
}
