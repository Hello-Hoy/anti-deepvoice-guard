package com.deepvoiceguard.app.ui.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * 하이브리드 모드 데이터 수집 동의 다이얼로그.
 * 3단계: 없음 / 메타데이터만 / 전체 (오디오 포함).
 */
@Composable
fun HybridConsentDialog(
    currentLevel: String,
    onConsent: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    var selected by remember { mutableStateOf(currentLevel) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text("데이터 수집 동의", fontWeight = FontWeight.Bold)
        },
        text = {
            Column(modifier = Modifier.fillMaxWidth()) {
                Text(
                    text = "서비스 품질 향상을 위해 탐지 데이터를 서버로 전송할 수 있습니다.",
                    style = MaterialTheme.typography.bodyMedium,
                )
                Spacer(modifier = Modifier.height(16.dp))

                ConsentOption(
                    label = "수집 안 함",
                    description = "순수 온디바이스 동작. 어떤 데이터도 외부로 전송되지 않습니다.",
                    selected = selected == "none",
                    onClick = { selected = "none" },
                )
                ConsentOption(
                    label = "메타데이터만",
                    description = "탐지 점수, 키워드, 위협 레벨만 익명으로 전송됩니다. 오디오는 전송되지 않습니다.",
                    selected = selected == "metadata",
                    onClick = { selected = "metadata" },
                )
                ConsentOption(
                    label = "전체 (오디오 포함)",
                    description = "오디오 세그먼트를 포함하여 전송됩니다. 모델 학습에 활용되며, 90일 후 자동 삭제됩니다.",
                    selected = selected == "full",
                    onClick = { selected = "full" },
                )
            }
        },
        confirmButton = {
            TextButton(onClick = { onConsent(selected) }) {
                Text("확인", fontWeight = FontWeight.Bold)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("취소")
            }
        },
    )
}

@Composable
private fun ConsentOption(
    label: String,
    description: String,
    selected: Boolean,
    onClick: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.Top,
    ) {
        RadioButton(selected = selected, onClick = onClick)
        Column(modifier = Modifier.padding(start = 8.dp)) {
            Text(text = label, fontWeight = FontWeight.SemiBold)
            Text(
                text = description,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
