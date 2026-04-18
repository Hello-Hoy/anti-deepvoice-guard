package com.deepvoiceguard.app.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.deepvoiceguard.app.storage.AppSettings
import com.deepvoiceguard.app.storage.SettingsRepository
import com.deepvoiceguard.app.stt.SttCapabilityChecker
import kotlinx.coroutines.launch

@Composable
fun SettingsScreen() {
    val context = LocalContext.current
    val repository = remember { SettingsRepository(context) }
    val settings by repository.settings.collectAsState(initial = AppSettings())
    val scope = rememberCoroutineScope()

    // STT 동의 다이얼로그 트리거 — 처음 STT를 켤 때만 표시.
    var showSttConsentDialog by remember { mutableStateOf(false) }
    // 하이브리드 모드 동의 다이얼로그 — opt-in 시만.
    var showHybridConsentDialog by remember { mutableStateOf(false) }

    // STT 가용성 — H94: verified on-device 가능 여부 기준 (API 31+ && on-device recognizer).
    val sttCapable = remember {
        runCatching {
            SttCapabilityChecker(context).isOfflineRecognitionSupported()
        }.getOrDefault(false)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(24.dp),
    ) {
        Text(
            "Settings",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
        )

        Spacer(modifier = Modifier.height(24.dp))

        // 추론 엔진 선택
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Inference Engine", style = MaterialTheme.typography.titleSmall)
                Spacer(modifier = Modifier.height(8.dp))

                SettingRow("On-Device (AASIST-L)", settings.useOnDevice) {
                    scope.launch { repository.setUseOnDevice(true) }
                }
                SettingRow("Server", !settings.useOnDevice) {
                    scope.launch { repository.setUseOnDevice(false) }
                }

                if (!settings.useOnDevice) {
                    Spacer(modifier = Modifier.height(8.dp))
                    OutlinedTextField(
                        value = settings.serverUrl,
                        onValueChange = { url ->
                            scope.launch { repository.setServerUrl(url) }
                        },
                        label = { Text("Server URL") },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true,
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // 민감도 설정
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Detection Sensitivity", style = MaterialTheme.typography.titleSmall)
                Spacer(modifier = Modifier.height(8.dp))
                Text("Threshold: ${String.format("%.2f", settings.threshold)}")
                Slider(
                    value = settings.threshold,
                    onValueChange = { value ->
                        scope.launch { repository.setThreshold(value) }
                    },
                    valueRange = 0.5f..0.95f,
                    steps = 8,
                )
                Text(
                    "Lower = more sensitive, higher = fewer false positives",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // 보이스피싱 탐지 (STT + 키워드) — 동의 필수.
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Phishing Detection (STT)", style = MaterialTheme.typography.titleSmall)
                Spacer(modifier = Modifier.height(8.dp))

                // STT 토글 — OFF→ON 전환 시 consent 없으면 다이얼로그 표시.
                val sttEffective = settings.sttEnabled && settings.sttConsentGiven
                SettingRow(
                    label = if (sttCapable) "Speech-to-Text (on-device)"
                            else "Speech-to-Text (on-device unavailable)",
                    checked = sttEffective,
                ) {
                    if (!sttCapable) return@SettingRow
                    if (sttEffective) {
                        // OFF 전환 — consent는 유지, enabled만 끈다.
                        scope.launch { repository.setSttEnabled(false) }
                    } else {
                        if (settings.sttConsentGiven) {
                            scope.launch { repository.setSttEnabled(true) }
                        } else {
                            showSttConsentDialog = true
                        }
                    }
                }
                Text(
                    if (sttCapable) {
                        "On-device SpeechRecognizer를 사용해 보이스피싱 키워드를 탐지합니다. " +
                            "음성은 기기 외부로 전송되지 않습니다 (Android 12 이상 필수)."
                    } else {
                        "이 기기에서는 on-device 음성 인식이 지원되지 않아 STT는 비활성화됩니다."
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                Spacer(modifier = Modifier.height(12.dp))

                SettingRow("Phishing keyword analysis", settings.phishingDetectionEnabled) {
                    scope.launch {
                        repository.setPhishingDetectionEnabled(!settings.phishingDetectionEnabled)
                    }
                }
                Text(
                    "STT 전사 결과에서 피싱 키워드/구문 패턴을 분석합니다.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                Spacer(modifier = Modifier.height(12.dp))

                SettingRow("Store transcripts locally", settings.transcriptStorageEnabled) {
                    scope.launch {
                        repository.setTranscriptStorageEnabled(!settings.transcriptStorageEnabled)
                    }
                }
                Text(
                    "전사 텍스트를 기기 내 History에 저장합니다 (7일 후 자동 삭제). 기본 OFF.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // 통화 모니터링 설정
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Call Monitoring", style = MaterialTheme.typography.titleSmall)
                Spacer(modifier = Modifier.height(8.dp))

                SettingRow("Auto-monitor calls", settings.autoStartOnCall) {
                    scope.launch { repository.setAutoStartOnCall(!settings.autoStartOnCall) }
                }
                Text(
                    "전화 수신/발신 시 자동으로 음성 모니터링을 시작합니다",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                Spacer(modifier = Modifier.height(12.dp))

                SettingRow("Auto-start on boot", settings.autoStartOnBoot) {
                    scope.launch { repository.setAutoStartOnBoot(!settings.autoStartOnBoot) }
                }
                Text(
                    "기기 재시작 시 자동으로 서비스를 시작합니다",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        // **H151**: Hybrid Mode UI는 runtime wiring이 완성될 때까지 hidden — dead config가
        // consent surface로 노출되어 사용자를 오도하지 않도록. 내부 hybridModeEnabled
        // /dataConsentLevel은 보존되지만 사용자 토글은 제거.

        // **H154**: Demo Mode toggle hide — DeepVoiceGuardApp가 settings.demoMode를 읽지 않으므로
        // dead config UI. asset 존재 여부로만 Demo tab이 활성화됨.

        Spacer(modifier = Modifier.height(16.dp))

        // 안내 카드
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.tertiaryContainer,
            ),
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("About Call Monitoring", style = MaterialTheme.typography.titleSmall)
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    "Android 보안 정책으로 인해 통화 중 상대방의 음성을 직접 캡처할 수 없습니다. " +
                    "스피커폰 모드에서는 마이크가 양쪽 음성을 모두 수집하므로 " +
                    "상대방 음성의 딥페이크 여부를 분석할 수 있습니다.\n\n" +
                    "최적의 탐지를 위해 통화 시 스피커폰 모드를 사용해주세요.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }

    if (showSttConsentDialog) {
        SttConsentDialog(
            onConsent = {
                scope.launch {
                    repository.setSttConsentGiven(true)
                    repository.setSttEnabled(true)
                }
                showSttConsentDialog = false
            },
            onDismiss = { showSttConsentDialog = false },
        )
    }

    if (showHybridConsentDialog) {
        HybridConsentDialog(
            currentLevel = settings.dataConsentLevel,
            onConsent = { level ->
                scope.launch {
                    repository.setDataConsentLevel(level)
                    repository.setHybridModeEnabled(level != "none")
                }
                showHybridConsentDialog = false
            },
            onDismiss = { showHybridConsentDialog = false },
        )
    }
}

@Composable
private fun SettingRow(label: String, checked: Boolean, onCheckedChange: () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label)
        Switch(checked = checked, onCheckedChange = { onCheckedChange() })
    }
}
