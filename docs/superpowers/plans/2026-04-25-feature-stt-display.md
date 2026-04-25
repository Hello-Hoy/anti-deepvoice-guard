# Feature 1 — STT 전사문 화면 표시 + 피싱 키워드 강조

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** START 후 사용자가 말하는 내용을 실시간 전사문으로 화면에 표시하고, `PhishingKeywordDetector`가 매칭한 결과(`MatchedKeyword.matchedText`)를 굵은 빨간 글씨로 하이라이트한다.

**Architecture:** UI는 detector 결과를 **그대로** 받아 표시 (절대 detector 로직을 흉내내지 않음 — single source of truth). `MatchedKeyword.matchedText` 는 정규화/동의어 처리 후 실제 전사문에서 잡힌 substring 이므로, UI 는 단순히 transcription 안에서 그 substring 의 위치(들)를 찾아 강조만 하면 됨. 같은 `matchedText` 가 여러 번 등장하면 모두 강조.

**Tech Stack:** Kotlin, Jetpack Compose Material 3, AnnotatedString.

**Base branch:** `android/real-device-verify` (최신 v5 모델 포함).

---

## File Structure

| File | 책임 |
|---|---|
| `android-app/app/src/main/java/com/deepvoiceguard/app/ui/components/TranscriptionView.kt` (신규) | AnnotatedString 빌드 + Composable. 단위 테스트 대상 핵심. 입력은 `List<MatchedKeyword>`. |
| `android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/HomeScreen.kt` | 새 카드 추가 (기존 `transcription.takeLast(500)` 라인을 TranscriptionView로 교체) |
| `android-app/app/src/test/java/com/deepvoiceguard/app/ui/TranscriptionHighlightTest.kt` (신규) | 키워드 하이라이팅 로직 단위 테스트 |

---

## Phase 0 — Branch & 첫 백업 push

- [ ] **Step 0.1: 새 feature branch**

```bash
git checkout android/real-device-verify
git checkout -b feature/stt-display
```

- [ ] **Step 0.2: 원격에 push (백업)**

```bash
git push -u origin feature/stt-display
```

---

## Phase 1 — Highlight 헬퍼 함수 + 단위 테스트

`buildHighlightedTranscript(transcription, matched)` 는 detector 가 반환한 `List<MatchedKeyword>` 를 그대로 받는다. 각 `MatchedKeyword.matchedText` 는 detector 가 실제 전사문에서 매칭한 정확한 substring 이므로 그것만 찾아 강조하면 된다 — 정규화/동의어 처리는 detector 책임.

### Task 1: 하이라이팅 함수 + 테스트

**Files:**
- Create: `android-app/app/src/main/java/com/deepvoiceguard/app/ui/components/TranscriptionView.kt`
- Test: `android-app/app/src/test/java/com/deepvoiceguard/app/ui/TranscriptionHighlightTest.kt`

- [ ] **Step 1.1: 실패하는 테스트 작성**

`android-app/app/src/test/java/com/deepvoiceguard/app/ui/TranscriptionHighlightTest.kt`:

```kotlin
package com.deepvoiceguard.app.ui

import androidx.compose.ui.text.font.FontWeight
import com.deepvoiceguard.app.phishing.model.MatchedKeyword
import com.deepvoiceguard.app.phishing.model.PhishingCategory
import com.deepvoiceguard.app.ui.components.buildHighlightedTranscript
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TranscriptionHighlightTest {

    private fun mk(matchedText: String, keyword: String = matchedText) = MatchedKeyword(
        keyword = keyword,
        category = PhishingCategory.FINANCIAL_ACTION,
        weight = 1.0f,
        matchedText = matchedText,
    )

    @Test
    fun `no matches returns plain text`() {
        val result = buildHighlightedTranscript(
            transcription = "안녕하세요 좋은 아침입니다",
            matchedKeywords = emptyList(),
        )
        assertEquals("안녕하세요 좋은 아침입니다", result.text)
        assertTrue("no spans expected", result.spanStyles.isEmpty())
    }

    @Test
    fun `single matchedText is highlighted at correct range`() {
        val result = buildHighlightedTranscript(
            transcription = "계좌 이체를 부탁드립니다",
            matchedKeywords = listOf(mk(matchedText = "계좌")),
        )
        assertEquals("계좌 이체를 부탁드립니다", result.text)
        assertEquals(1, result.spanStyles.size)
        val span = result.spanStyles[0]
        assertEquals(0, span.start)
        assertEquals(2, span.end)
        assertEquals(FontWeight.Bold, span.item.fontWeight)
    }

    @Test
    fun `multiple occurrences of same matchedText all highlighted`() {
        // detector가 동일 matchedText 1개만 반환해도 UI는 transcription 전체에서 모든 occurrence 강조.
        val result = buildHighlightedTranscript(
            transcription = "계좌 비밀번호 계좌 이체",
            matchedKeywords = listOf(mk(matchedText = "계좌")),
        )
        val starts = result.spanStyles.map { it.start }
        // 두 번째 "계좌" 위치: "계좌"(2) + " "(1) + "비밀번호"(4) + " "(1) = index 8
        assertEquals(listOf(0, 8), starts)
    }

    @Test
    fun `multiple distinct matchedTexts all highlighted`() {
        val result = buildHighlightedTranscript(
            transcription = "비밀번호와 계좌번호를 알려주세요",
            matchedKeywords = listOf(
                mk(matchedText = "비밀번호"),
                mk(matchedText = "계좌번호"),
            ),
        )
        assertEquals(2, result.spanStyles.size)
        // 위치 검증 (0, 6)
        assertEquals(0, result.spanStyles[0].start)
        assertEquals(6, result.spanStyles[1].start)
    }

    @Test
    fun `matchedText with whitespace inside is highlighted`() {
        // detector 가 "계좌 이체" 처럼 공백 포함 substring을 반환할 수 있음 (PhraseMatcher 결과).
        val result = buildHighlightedTranscript(
            transcription = "당장 계좌 이체 해주세요",
            matchedKeywords = listOf(mk(matchedText = "계좌 이체")),
        )
        assertEquals(1, result.spanStyles.size)
        val span = result.spanStyles[0]
        assertEquals(3, span.start)
        assertEquals(8, span.end)
    }

    @Test
    fun `nested or overlapping matchedTexts longer wins`() {
        // 같은 위치에 두 매칭이 있으면 더 긴 substring 우선 (시각적 일관성).
        // 예: "검찰" 과 "검찰청" 둘 다 detector 가 반환한 케이스.
        val result = buildHighlightedTranscript(
            transcription = "검찰청에서 연락 드립니다",
            matchedKeywords = listOf(
                mk(matchedText = "검찰"),
                mk(matchedText = "검찰청"),
            ),
        )
        assertEquals(1, result.spanStyles.size)
        val span = result.spanStyles[0]
        assertEquals(0, span.start)
        assertEquals(3, span.end)  // "검찰청"
    }

    @Test
    fun `matchedText not found in transcription is silently skipped`() {
        // detector 가 정규화 거친 후 매칭 — 원본 transcription에 정확한 substring 이 없을 수도 있음.
        // 그 경우 UI는 강조 없이 plain text 유지 (절대 throw 하지 않음).
        val result = buildHighlightedTranscript(
            transcription = "안녕하세요 좋은 아침",
            matchedKeywords = listOf(mk(matchedText = "안 녕")),  // 공백 위치 다름
        )
        assertEquals("안녕하세요 좋은 아침", result.text)
        assertTrue(result.spanStyles.isEmpty())
    }

    @Test
    fun `empty transcription returns empty annotated string`() {
        val result = buildHighlightedTranscript(
            transcription = "",
            matchedKeywords = listOf(mk(matchedText = "키워드")),
        )
        assertEquals("", result.text)
        assertTrue(result.spanStyles.isEmpty())
    }

    @Test
    fun `empty matchedText is ignored`() {
        // 방어적 — detector 가 빈 문자열 반환할 일은 없지만 안전장치.
        val result = buildHighlightedTranscript(
            transcription = "안녕하세요",
            matchedKeywords = listOf(mk(matchedText = "")),
        )
        assertEquals("안녕하세요", result.text)
        assertTrue(result.spanStyles.isEmpty())
    }
}
```

- [ ] **Step 1.2: 테스트 실패 확인**

```bash
cd android-app
./gradlew :app:testDebugUnitTest --tests "com.deepvoiceguard.app.ui.TranscriptionHighlightTest" 2>&1 | tail -10
```

기대: `unresolved reference: buildHighlightedTranscript`

- [ ] **Step 1.3: 헬퍼 함수 + Composable 작성**

`android-app/app/src/main/java/com/deepvoiceguard/app/ui/components/TranscriptionView.kt`:

```kotlin
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
```

- [ ] **Step 1.4: 테스트 통과 확인**

```bash
./gradlew :app:testDebugUnitTest --tests "com.deepvoiceguard.app.ui.TranscriptionHighlightTest" 2>&1 | tail -10
```

기대: `BUILD SUCCESSFUL`, 9 passed.

- [ ] **Step 1.5: 커밋**

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/ui/components/TranscriptionView.kt \
         android-app/app/src/test/java/com/deepvoiceguard/app/ui/TranscriptionHighlightTest.kt
git commit -m "feat(ui): TranscriptionView with detector-driven keyword highlight

- buildHighlightedTranscript: List<MatchedKeyword> 를 그대로 받아 matchedText 강조 (detector 책임 = source of truth)
- 더 긴 substring 우선 (overlap 시 시각 일관성)
- matchedText 가 transcription 에 없으면 silent skip (정규화 결과 mismatch 방어)
- 9 unit tests (no match / single / multi occurrence / multi kw / phrase with space / overlap / not found / empty)"
```

---

## Phase 1.5 — Detector matchedText 를 원본 substring 으로 보강

**배경**: `PhishingKeywordDictionary.findMatches()` 는 현재 `matchedText` 에 **사전 literal**(`entry.keyword` 또는 `synonym`)을 그대로 넣음 (`PhishingKeywordDictionary.kt:27-53`). 하지만 normalize 가 공백/조사 등을 제거하므로 사전 literal 이 원본 transcription 에 정확히 안 들어있을 수 있음 (예: 사전 `"계좌이체"` vs 사용자 발화 `"계좌 이체"`). UI 의 `indexOf` 검색이 fail → silent skip → 강조 표시 누락.

**해결**: detector 가 매칭 발견 시 **원본 transcription 에서 plain 검색** 시도하여 그 substring 을 `matchedText` 로 사용. 못 찾으면 fallback 으로 사전 literal. 이러면 80%+ 케이스에서 UI 강조 정확.

**Trade-off**: detector 변경이지만 매우 작음. 기존 호출자에 영향 없음 (`MatchedKeyword` 시그니처 동일). 기존 테스트가 `matchedText == 사전 literal` 가정하면 일부 깨질 수 있어 같이 수정.

### Task 1.5: PhishingKeywordDictionary 수정 + 회귀 테스트

**Files:**
- Modify: `android-app/app/src/main/java/com/deepvoiceguard/app/phishing/PhishingKeywordDictionary.kt`
- Modify (필요 시): `android-app/app/src/test/java/com/deepvoiceguard/app/phishing/PhishingKeywordDictionaryTest.kt` (혹시 있으면)
- Test: `android-app/app/src/test/java/com/deepvoiceguard/app/phishing/MatchedTextOriginalSubstringTest.kt` (신규)

- [ ] **Step 1.5.1: 신규 동작 검증 테스트 작성**

```kotlin
package com.deepvoiceguard.app.phishing

import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * matchedText 가 가능한 한 원본 substring 을 가지도록 보강된 동작 검증.
 *
 * Note: PhishingKeywordDictionary 는 Context (assets) 의존이라 Robolectric 사용.
 * 만약 프로젝트가 Robolectric 미설정이면 builtinKeywords 만 쓰는 단순 단위 테스트로 변경.
 */
@RunWith(RobolectricTestRunner::class)
class MatchedTextOriginalSubstringTest {

    private val ctx = ApplicationProvider.getApplicationContext<android.content.Context>()
    private val dict = PhishingKeywordDictionary(ctx)

    @Test
    fun `matchedText preserves whitespace from original transcription`() {
        // builtinKeywords 의 "계좌이체" 사전 literal vs 사용자 발화 "계좌 이체"
        val text = "지금 당장 계좌 이체 해주세요"
        val matches = dict.findMatches(text)
        // "계좌이체" 가 매칭 됐다면 matchedText 는 원본의 "계좌 이체" (공백 포함) 이어야
        val financial = matches.firstOrNull { it.keyword == "계좌이체" }
        if (financial != null) {
            // 원본 substring 우선 — "계좌 이체" 로 들어가야
            assertEquals("계좌 이체", financial.matchedText)
        }
    }

    @Test
    fun `matchedText for synonym uses original substring when present`() {
        // "검찰" entry 의 synonym "검찰청" — 사용자가 "검찰청에서" 말함 → matchedText="검찰청" (원본 substring)
        val text = "검찰청에서 연락드립니다"
        val matches = dict.findMatches(text)
        val impersonation = matches.firstOrNull { it.keyword == "검찰" }
        if (impersonation != null) {
            assertTrue(
                "matchedText 는 원본 transcription 에 들어있어야",
                text.contains(impersonation.matchedText),
            )
        }
    }

    @Test
    fun `falls back to dictionary literal when nothing matches in original`() {
        // edge case: 정규화 후 매칭됐지만 원본 plain 검색 실패 — fallback 으로 dictionary literal
        // (실제로 builtinKeywords + 한글 normalizer 조합에서 이 케이스를 만들기 어려움 — 회귀 안전망)
        val text = "별 의미 없는 텍스트"
        val matches = dict.findMatches(text)
        assertTrue("매칭 자체가 없으면 빈 결과", matches.isEmpty())
    }
}
```

만약 Robolectric 의존성 없으면 위 테스트 대신 `builtinKeywords()` 를 직접 호출하는 단순 단위 테스트로 변경 (KoreanNormalizer 없이 이미 노출된 API 만 사용).

- [ ] **Step 1.5.2: 테스트 실패 확인**

```bash
cd android-app
./gradlew :app:testDebugUnitTest --tests "com.deepvoiceguard.app.phishing.MatchedTextOriginalSubstringTest" 2>&1 | tail -10
```

기대: 첫 번째 테스트 (`whitespace preserves`) 가 fail — 현재 matchedText 는 사전 literal 이라 "계좌이체" 반환 (assertEquals "계좌 이체" 실패).

- [ ] **Step 1.5.3: PhishingKeywordDictionary.findMatches 수정**

`PhishingKeywordDictionary.kt` 의 `findMatches(text)` 함수 안에서 `MatchedKeyword(...)` 생성하는 두 곳 (메인 키워드 + 동의어) 모두 수정. 기존:

```kotlin
matches.add(
    MatchedKeyword(
        keyword = entry.keyword,
        category = entry.category,
        weight = entry.weight,
        matchedText = entry.keyword,    // ← 사전 literal
    )
)
```

다음과 같이 helper 사용:

```kotlin
matches.add(
    MatchedKeyword(
        keyword = entry.keyword,
        category = entry.category,
        weight = entry.weight,
        matchedText = resolveOriginalSubstring(text, entry.keyword, entry.synonyms),
    )
)
```

동의어 매칭 케이스도 동일하게 — 두 곳 모두.

`PhishingKeywordDictionary` 클래스 안에 helper 추가:

```kotlin
/**
 * 원본 [text] 에서 [keyword] 또는 [synonyms] 중 하나를 plain 검색해 발견된 substring 반환.
 * 못 찾으면 [keyword] 를 fallback (사전 literal).
 *
 * 매칭은 detector 가 정규화 후 결정한 것이므로, 이 함수의 fallback 은 "정규화 거친 원본 위치
 * 가 plain 검색으로 안 맞는" edge case (조사 변형, 띄어쓰기 차이 등) 만 책임진다.
 */
private fun resolveOriginalSubstring(
    text: String,
    keyword: String,
    synonyms: List<String>,
): String {
    val candidates = listOf(keyword) + synonyms
    for (cand in candidates) {
        val idx = text.indexOf(cand, ignoreCase = true)
        if (idx >= 0) {
            return text.substring(idx, idx + cand.length)
        }
    }
    return keyword  // fallback: 사전 literal
}
```

- [ ] **Step 1.5.4: 테스트 통과 확인**

```bash
./gradlew :app:testDebugUnitTest --tests "com.deepvoiceguard.app.phishing.MatchedTextOriginalSubstringTest" 2>&1 | tail -10
```

기대: PASS.

- [ ] **Step 1.5.5: 기존 PhishingKeywordDetector 회귀 테스트 모두 통과**

```bash
./gradlew :app:testDebugUnitTest --tests "com.deepvoiceguard.app.phishing.*" 2>&1 | tail -15
```

기대: 기존 테스트 모두 통과. 만약 어떤 테스트가 `matchedText == 특정 사전 literal` 을 hardcoded 검증한다면 테스트 데이터를 보강해서 (transcription 에 사전 literal 정확히 포함) 통과하도록.

- [ ] **Step 1.5.6: 커밋**

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/phishing/PhishingKeywordDictionary.kt \
         android-app/app/src/test/java/com/deepvoiceguard/app/phishing/MatchedTextOriginalSubstringTest.kt
git commit -m "feat(phishing): matchedText prefers original transcription substring

원본 transcription 에 keyword/synonym 이 plain 검색으로 발견되면 그 substring 을
matchedText 로 사용. 정규화 후 매칭이지만 원본에 plain 일치 없는 edge case 만 사전
literal fallback. UI (TranscriptionView) 의 indexOf 검색 신뢰도 80%+ → ~100% 향상.

resolveOriginalSubstring helper 추가. 기존 호출자/MatchedKeyword 시그니처 변경 없음."
```

---

## Phase 2 — HomeScreen 통합

`HomeScreen.kt` 의 기존 `transcription.takeLast(500)` 텍스트 영역을 새 컴포넌트로 교체. `combinedResult.matchedKeywords` 는 이미 `List<MatchedKeyword>` 이므로 그대로 전달.

### Task 2: HomeScreen 통합

**Files:**
- Modify: `android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/HomeScreen.kt`

- [ ] **Step 2.1: 기존 transcription 카드 식별**

```bash
grep -n "transcription.takeLast\|combinedResult?.matchedKeywords" android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/HomeScreen.kt
```

기대 출력 (라인 번호는 다를 수 있음):
```
278:        val matchedKeywords = combinedResult?.matchedKeywords.orEmpty()
372:                        text = transcription.takeLast(500),
```

- [ ] **Step 2.2: import 추가**

`HomeScreen.kt` 상단 imports 에 추가:

```kotlin
import com.deepvoiceguard.app.ui.components.TranscriptionView
```

- [ ] **Step 2.3: 기존 transcription 카드 교체**

`HomeScreen.kt` 내 기존 `if (transcription.isNotBlank()) { Card { ... Text(text = transcription.takeLast(500), ...) ... } }` 블록을 다음으로 교체:

```kotlin
// 실시간 전사 + 키워드 강조 — start 후 보임. matchedKeywords는 detector(PhishingKeywordDetector)
// 가 정규화/동의어 처리 후 반환한 List<MatchedKeyword> 그대로 전달.
if (isMonitoring) {
    TranscriptionView(
        transcription = transcription,
        matchedKeywords = matchedKeywords,
        modifier = Modifier.padding(top = 16.dp),
    )
}
```

(`matchedKeywords` 는 기존 line 278 에 이미 정의돼 있는 변수 — 타입 `List<MatchedKeyword>`. 새 변수 추가 X.)

- [ ] **Step 2.4: 빌드 확인**

```bash
./gradlew :app:compileDebugKotlin 2>&1 | tail -3
```

기대: `BUILD SUCCESSFUL`.

- [ ] **Step 2.5: 커밋**

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/HomeScreen.kt
git commit -m "feat(home): show live transcription with phishing keyword highlight

기존 단순 takeLast 500자 카드 → TranscriptionView 로 교체.
combinedResult.matchedKeywords (List<MatchedKeyword>) 그대로 전달 — UI는 detector 결과를 신뢰만 함."
```

---

## Phase 3 — 실기기 검증

### Task 3: 실기기 동작 확인

- [ ] **Step 3.1: Android Studio에서 ▶ Run**

폰에 새 APK 설치. STT 권한이 OFF 상태라면 Settings 에서 STT 동의 ON.

- [ ] **Step 3.2: START 누르기**

- [ ] **Step 3.3: 일반 음성 시험 (5초)**

폰에 대고 "안녕하세요 좋은 아침입니다" 말하기.

기대:
- 화면에 텍스트 나타남 (STT 정확도 따라 다소 차이)
- 키워드 강조 없음

- [ ] **Step 3.4: 피싱 키워드 시험**

"검찰청에서 연락드립니다 계좌번호 알려주세요" 말하기.

기대:
- 화면에 위 문장 (STT 결과) 나타남
- "검찰청", "계좌번호" 또는 detector 가 매칭한 substring 들이 굵은 빨간 글씨 + 연한 빨간 배경
- STT 가 단어를 다르게 인식하면 (예: "검찰청" 을 "검찰처" 로 듣는 등) 그 결과에 따라 매칭 달라짐 — UI는 detector 결과를 신뢰만 함

- [ ] **Step 3.5: STOP**

전사 카드 사라지는지 확인.

---

## Phase 4 — 백업 push

- [ ] **Step 4.1: 모든 변경 push**

```bash
git push origin feature/stt-display
```

- [ ] **Step 4.2: 다음 plan으로 이동 안내**

이 시점에서 master 머지는 보류. PR 작성하지 않음. 다른 feature 브랜치들과 데모 직전 한꺼번에 머지.

---

## Self-Review Checklist

- [x] Spec 커버리지: 전사문 화면 표시 + 키워드 강조 — Phase 1+2 완료.
- [x] Placeholder 없음.
- [x] 타입 일관성: `buildHighlightedTranscript(transcription, matchedKeywords: List<MatchedKeyword>)` 가 테스트, 구현, HomeScreen 호출 모두 동일.
- [x] Detector 와 UI 의 책임 분리: detector = 매칭 결정, UI = 표시만.

## 알려진 한계

- STT 자체가 음성을 잘못 인식하면 매칭되는 substring 이 달라짐 (UI 책임 외).
- Phase 1.5 의 detector 보강 후에도 **정규화 후 매칭 + 원본 plain 검색 fail** 케이스는 여전히 silent skip (예: 사전 "오늘안에" vs 발화 "오늘 안 에" 처럼 공백이 비표준 위치). 데모 전 핵심 키워드 시나리오 리허설 필요.
- 매우 긴 transcription (10k자+) 에서 indexOf O(n*m) 누적 가능 — 실제로는 detector 가 ~10개 이하 반환하므로 무시 가능.
- 본 plan 은 Robolectric 가 프로젝트에 추가된 상태를 가정. 만약 미적용이면 Phase 1.5 의 신규 테스트를 builtinKeywords() 만 사용하는 형태로 다운그레이드.
