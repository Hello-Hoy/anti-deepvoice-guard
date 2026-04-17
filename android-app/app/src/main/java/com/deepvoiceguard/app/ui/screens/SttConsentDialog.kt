package com.deepvoiceguard.app.ui.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * STT 동의 다이얼로그.
 * 앱은 on-device recognizer가 검증된 기기에서만 STT를 활성화한다.
 */
@Composable
fun SttConsentDialog(
    onConsent: () -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(
                text = "음성 인식 동의",
                fontWeight = FontWeight.Bold,
            )
        },
        text = {
            Column(modifier = Modifier.fillMaxWidth()) {
                Text(
                    text = "보이스피싱 키워드 탐지를 위해 음성 인식(STT) 기능을 활성화합니다.",
                    style = MaterialTheme.typography.bodyMedium,
                )
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text = "안내사항:",
                    fontWeight = FontWeight.SemiBold,
                    style = MaterialTheme.typography.bodyMedium,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "• 개인정보 보호를 위해 음성 인식은 **on-device recognizer**로만 처리됩니다\n" +
                            "• 한국어 오프라인 음성 모델이 설치되어 있어야 동작합니다\n" +
                            "• 오프라인 모델이 없으면 STT는 자동 비활성화됩니다\n" +
                            "• 음성 데이터는 앱에 저장되지 않습니다\n" +
                            "• 설정에서 언제든 비활성화할 수 있습니다",
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(start = 8.dp),
                )
            }
        },
        confirmButton = {
            TextButton(onClick = onConsent) {
                Text("동의", fontWeight = FontWeight.Bold)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("거부")
            }
        },
    )
}
