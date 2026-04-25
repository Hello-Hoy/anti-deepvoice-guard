package com.deepvoiceguard.app.ui.components

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import com.deepvoiceguard.app.phishing.model.MatchedKeyword

/**
 * 전사문과 detector가 반환한 [MatchedKeyword] 리스트를 받아, 각 `matchedText` 의
 * 모든 occurrence 위치에 굵은 빨강 강조를 입힌 [AnnotatedString] 을 빌드한다.
 *
 * **설계 원칙**: detector 의 매칭 로직(정규화/동의어/조사 처리 등)은 단일 source of truth.
 * UI 는 detector 가 이미 결정한 `matchedText` 만 그대로 찾아 표시한다. 정규화 처리 결과로
 * `matchedText` 가 원본 transcription 에 정확히 안 들어있으면 그 항목은 건너뛴다 (throw X).
 *
 * 같은 위치에 여러 매칭이 겹치면 더 긴 substring 이 우선 (시각적 일관성).
 */
fun buildHighlightedTranscript(
    transcription: String,
    matchedKeywords: List<MatchedKeyword>,
): AnnotatedString {
    if (transcription.isEmpty()) {
        return AnnotatedString("")
    }

    // 모든 substring occurrence 위치 수집. 더 긴 매칭 우선.
    val texts = matchedKeywords
        .map { it.matchedText }
        .filter { it.isNotEmpty() }
        .distinct()
        .sortedByDescending { it.length }

    val matched = mutableListOf<IntRange>()
    for (text in texts) {
        var index = 0
        while (true) {
            val found = transcription.indexOf(text, startIndex = index)
            if (found < 0) break
            val range = found until (found + text.length)
            // 이미 매칭된 더 긴 범위에 포함되면 skip (overlap → 긴 매칭 우선)
            val isContainedInExisting = matched.any {
                it.first <= range.first && range.last <= it.last
            }
            if (!isContainedInExisting) matched.add(range)
            index = found + 1
        }
    }
    val sortedRanges = matched.sortedBy { it.first }

    return buildAnnotatedString {
        var cursor = 0
        for (range in sortedRanges) {
            if (range.first < cursor) continue
            if (range.first > cursor) {
                append(transcription.substring(cursor, range.first))
            }
            withStyle(
                SpanStyle(
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFFD32F2F),  // material red 700
                    background = Color(0x33FF5252),
                )
            ) {
                append(transcription.substring(range.first, range.last + 1))
            }
            cursor = range.last + 1
        }
        if (cursor < transcription.length) {
            append(transcription.substring(cursor))
        }
    }
}

@Composable
fun TranscriptionView(
    transcription: String,
    matchedKeywords: List<MatchedKeyword>,
    modifier: Modifier = Modifier,
) {
    val annotated = buildHighlightedTranscript(transcription, matchedKeywords)

    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                "실시간 전사",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
            )
            Spacer(modifier = Modifier.height(8.dp))
            if (annotated.text.isBlank()) {
                Text(
                    "(아직 인식된 음성 없음)",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                Column(
                    modifier = Modifier
                        .heightIn(max = 200.dp)
                        .verticalScroll(rememberScrollState()),
                ) {
                    Text(
                        text = annotated,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
        }
    }
}
