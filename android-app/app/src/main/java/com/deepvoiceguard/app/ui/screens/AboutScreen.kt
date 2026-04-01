package com.deepvoiceguard.app.ui.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

@Composable
fun AboutScreen() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
    ) {
        Text(
            "About",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
        )
        Spacer(modifier = Modifier.height(16.dp))
        InfoRow("App", "Anti-DeepVoice Guard v1.0.0")
        InfoRow("Model", "AASIST (ONNX)")
        InfoRow("Model Size", "~0.6 MB")
        InfoRow("Input", "~4s @ 16kHz (64,600 samples)")
        InfoRow("VAD", "Silero VAD v5")
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            "Based on AASIST (Audio Anti-Spoofing using Integrated " +
            "Spectro-Temporal Graph Attention Networks)",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Text("$label: $value", style = MaterialTheme.typography.bodyMedium)
    Spacer(modifier = Modifier.height(4.dp))
}
