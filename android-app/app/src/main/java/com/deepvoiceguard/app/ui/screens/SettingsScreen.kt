package com.deepvoiceguard.app.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
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
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.deepvoiceguard.app.storage.AppSettings
import com.deepvoiceguard.app.storage.SettingsRepository
import kotlinx.coroutines.launch

@Composable
fun SettingsScreen() {
    val context = LocalContext.current
    val repository = remember { SettingsRepository(context) }
    val settings by repository.settings.collectAsState(initial = AppSettings())
    val scope = rememberCoroutineScope()

    Column(
        modifier = Modifier
            .fillMaxSize()
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
