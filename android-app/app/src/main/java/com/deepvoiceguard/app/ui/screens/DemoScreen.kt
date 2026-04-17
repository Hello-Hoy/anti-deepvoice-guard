package com.deepvoiceguard.app.ui.screens

import android.content.res.AssetFileDescriptor
import android.media.AudioAttributes
import android.media.MediaPlayer
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ElevatedButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import com.deepvoiceguard.app.inference.CombinedThreatLevel
import com.deepvoiceguard.app.inference.DemoAnalysisException
import com.deepvoiceguard.app.inference.DemoAnalysisPipeline
import com.deepvoiceguard.app.inference.DemoResult
import com.deepvoiceguard.app.inference.DemoScenario
import com.deepvoiceguard.app.inference.OnDeviceEngine
import com.deepvoiceguard.app.phishing.PhishingKeywordDetector
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.min

private val demoScenarios = listOf(
    DemoScenario(1, "일상 통화", "실제 사람의 일상 대화", "demo/demo_01.wav", "demo/demo_01_transcript.txt", "SAFE"),
    DemoScenario(2, "TTS 일상 대화", "AI 생성 음성 (일상 대화)", "demo/demo_02.wav", "demo/demo_02_transcript.txt", "DANGER"),
    DemoScenario(3, "실제 사람 피싱", "사람이 읽는 피싱 스크립트", "demo/demo_03.wav", "demo/demo_03_transcript.txt", "WARNING"),
    DemoScenario(4, "TTS 피싱", "AI 생성 음성 + 피싱 스크립트", "demo/demo_04.wav", "demo/demo_04_transcript.txt", "CRITICAL"),
    DemoScenario(5, "은행 정상 안내", "정상적인 은행 확인 전화", "demo/demo_05.wav", "demo/demo_05_transcript.txt", "SAFE"),
    DemoScenario(6, "보험 사기", "보험금 사기 시도", "demo/demo_06.wav", "demo/demo_06_transcript.txt", "WARNING"),
)

private const val WAVEFORM_BUCKETS = 60
private const val TYPING_INTERVAL_MS = 35L
private const val PLAYBACK_TICK_MS = 60L

private enum class DemoPhase { IDLE, PLAYING, REVEALING, DONE }

@Composable
fun DemoScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var analysisResult by remember { mutableStateOf<DemoResult?>(null) }
    var selectedScenario by remember { mutableStateOf<DemoScenario?>(null) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var phase by remember { mutableStateOf(DemoPhase.IDLE) }
    var playbackProgress by remember { mutableStateOf(0f) }
    var waveform by remember { mutableStateOf(FloatArray(WAVEFORM_BUCKETS)) }
    var transcriptShown by remember { mutableStateOf("") }

    // 엔진 초기화는 asset/ONNX 문제 시 예외를 던질 수 있으므로 구성 시 안전 생성.
    var initError by remember { mutableStateOf<String?>(null) }
    val engine = remember(context) {
        runCatching { OnDeviceEngine(context) }
            .onFailure { initError = "추론 엔진 초기화 실패: ${it.message}" }
            .getOrNull()
    }
    val detector = remember(context) { PhishingKeywordDetector(context) }
    val pipeline = remember(context, engine, detector) {
        engine?.let { DemoAnalysisPipeline(context, it, detector) }
    }

    // MediaPlayer + ticker job은 DisposableEffect로 생명주기 관리.
    var mediaPlayer by remember { mutableStateOf<MediaPlayer?>(null) }
    var tickerJob by remember { mutableStateOf<Job?>(null) }
    var typingJob by remember { mutableStateOf<Job?>(null) }
    var scenarioJob by remember { mutableStateOf<Job?>(null) }
    // 연속 탭에서도 시나리오 본문이 엄격히 직렬로 실행되도록 보장하는 뮤텍스.
    val scenarioMutex = remember { Mutex() }
    // 이전 시나리오의 뒤늦은 write가 새 시나리오 UI를 오염시키지 않도록 세대 토큰.
    val scenarioGenerationRef = remember { AtomicLong(0L) }

    DisposableEffect(engine) {
        onDispose {
            tickerJob?.cancel()
            typingJob?.cancel()
            val running = scenarioJob
            val mpOrphan = if (running == null) mediaPlayer else null
            val engineRef = engine
            if (running == null) {
                mpOrphan?.let {
                    runCatching { if (it.isPlaying) it.stop() }
                    runCatching { it.release() }
                }
                engineRef?.let { runCatching { it.close() } }
            } else {
                val cleanupScope = kotlinx.coroutines.MainScope()
                cleanupScope.launch {
                    try {
                        running.cancelAndJoin()
                        engineRef?.let { runCatching { it.close() } }
                    } finally {
                        cleanupScope.cancel()
                    }
                }
            }
        }
    }

    // 엔진/파이프라인 초기화 실패 시 화면 전체를 에러 상태로 치환.
    if (pipeline == null) {
        Column(
            modifier = Modifier.fillMaxSize().padding(16.dp),
        ) {
            Text(
                text = "Demo Mode",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = initError ?: "엔진 초기화 실패",
                color = MaterialTheme.colorScheme.error,
            )
        }
        return
    }

    data class AssetStatus(val audioPlayable: Boolean, val transcriptAvailable: Boolean) {
        val available: Boolean get() = audioPlayable && transcriptAvailable
    }
    val scenarioAvailability = remember {
        demoScenarios.associate { scenario ->
            val openable = runCatching {
                context.assets.openFd(scenario.audioAsset).use { true }
            }.getOrDefault(false)
            val transcript = pipeline.assetExists(scenario.transcriptAsset)
            scenario.id to AssetStatus(openable, transcript)
        }
    }
    val missingAudioIds = scenarioAvailability.filterValues { !it.audioPlayable }.keys
    val missingTranscriptIds = scenarioAvailability
        .filterValues { it.audioPlayable && !it.transcriptAvailable }
        .keys

    fun resetScreen() {
        // 세대 번호를 먼저 증가시켜 모든 stillMine() 가드가 즉시 false가 되도록 한다.
        scenarioGenerationRef.incrementAndGet()
        // scenarioJob에는 cancel만 호출하고 참조는 유지 — onDispose가 아직 finalize 중인 job을
        // 감지하고 cancelAndJoin 경로로 정리해 engine.close() 경쟁을 피한다.
        scenarioJob?.cancel()
        tickerJob?.cancel(); tickerJob = null
        typingJob?.cancel(); typingJob = null
        mediaPlayer?.let {
            runCatching { if (it.isPlaying) it.stop() }
            runCatching { it.release() }
        }
        mediaPlayer = null
        analysisResult = null
        selectedScenario = null
        errorMessage = null
        phase = DemoPhase.IDLE
        playbackProgress = 0f
        waveform = FloatArray(WAVEFORM_BUCKETS)
        transcriptShown = ""
    }

    fun startScenario(scenario: DemoScenario) {
        // 이전 작업에 취소 신호 전송. Mutex가 A→B→C 체인에서도 본문 실행을
        // 직렬화하므로 "중간 job 스킵" 경쟁이 발생하지 않는다.
        scenarioJob?.cancel()
        val myGen = scenarioGenerationRef.incrementAndGet()
        selectedScenario = scenario
        errorMessage = null
        analysisResult = null
        playbackProgress = 0f
        transcriptShown = ""
        waveform = FloatArray(WAVEFORM_BUCKETS)
        phase = DemoPhase.PLAYING

        // generation 일치일 때만 UI state를 쓰는 가드.
        fun stillMine() = scenarioGenerationRef.get() == myGen

        scenarioJob = scope.launch {
            scenarioMutex.withLock {
            val waveformDeferred: Deferred<FloatArray> = async(Dispatchers.IO) {
                loadWaveform(context, scenario.audioAsset)
            }
            val analysisDeferred: Deferred<Result<DemoResult>> = async(Dispatchers.IO) {
                runCatching { pipeline.analyze(scenario.audioAsset, scenario.transcriptAsset) }
            }
            val playbackDeferred: Deferred<Boolean> = async {
                startPlayback(
                    context = context,
                    scope = this,
                    assetPath = scenario.audioAsset,
                    onPrepared = { mp ->
                        if (stillMine()) {
                            mediaPlayer?.let { runCatching { it.release() } }
                            mediaPlayer = mp
                        } else {
                            // 취소된 시나리오의 준비된 player는 현재 UI 상태를 건드리지 않고 해제.
                            runCatching { mp.release() }
                        }
                    },
                    onReleased = {
                        if (stillMine()) mediaPlayer = null
                    },
                    onTick = { ratio ->
                        if (stillMine()) playbackProgress = ratio
                    },
                    attachTicker = { job ->
                        if (stillMine()) {
                            tickerJob?.cancel()
                            tickerJob = job
                        } else {
                            job.cancel()
                        }
                    },
                )
            }

            // 분석 결과를 먼저 기다려 실패를 즉시 노출. 실패 시 형제 job을 취소한다.
            val analysisOutcome = analysisDeferred.await()
            // 취소된 이전 시나리오가 새 시나리오의 상태를 덮지 않도록 각 suspension 지점에서 활성 검사.
            ensureActive()
            if (analysisOutcome.isFailure) {
                playbackDeferred.cancel()
                waveformDeferred.cancel()
                val e = analysisOutcome.exceptionOrNull()
                if (stillMine()) {
                    val reason = (e as? DemoAnalysisException)?.reason
                    errorMessage = if (reason != null) "분석 실패($reason): ${e.message}" else "분석 실패: ${e?.message ?: "unknown"}"
                    phase = DemoPhase.IDLE
                    selectedScenario = null
                }
                return@launch
            }

            val result = analysisOutcome.getOrThrow()
            ensureActive()
            if (!stillMine()) return@launch
            analysisResult = result
            // 파형/재생 await — CancellationException은 rethrow, 그 외 실패는 기본값으로 진행.
            // runCatching은 CancellationException도 Result.failure로 포장하여 상위 활성 검사를 우회하므로 사용 금지.
            val waveformResult = try {
                waveformDeferred.await()
            } catch (ce: kotlinx.coroutines.CancellationException) {
                throw ce
            } catch (_: Throwable) {
                FloatArray(WAVEFORM_BUCKETS)
            }
            ensureActive()
            if (!stillMine()) return@launch
            waveform = waveformResult

            val playbackOk = try {
                playbackDeferred.await()
            } catch (ce: kotlinx.coroutines.CancellationException) {
                throw ce
            } catch (_: Throwable) {
                false
            }
            ensureActive()
            if (!stillMine()) return@launch

            phase = DemoPhase.REVEALING
            // 재생 성공 시에만 progress bar를 완료로 채운다. 실패했다면 실제 멈춘 지점을 유지.
            if (playbackOk) playbackProgress = 1f
            typingJob?.cancel()
            typingJob = launch {
                typeTranscript(result.transcript) { chunk ->
                    // 본 scenario가 여전히 현재 세대일 때만 UI에 반영.
                    if (stillMine()) transcriptShown = chunk
                }
            }
            typingJob?.join()
            ensureActive()
            if (!stillMine()) return@launch
            phase = DemoPhase.DONE
            if (!playbackOk) {
                errorMessage = "오디오 재생에 실패했으나 분석은 완료됨"
            }
            } // scenarioMutex.withLock 종료
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
    ) {
        Text(
            text = "Demo Mode",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
        )
        Text(
            text = "프리셋 시나리오를 선택하여 분석합니다",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(modifier = Modifier.height(16.dp))

        val currentScenario = selectedScenario
        if (currentScenario != null && phase != DemoPhase.IDLE) {
            DemoPlaybackSection(
                scenario = currentScenario,
                phase = phase,
                waveform = waveform,
                progress = playbackProgress,
            )
            Spacer(modifier = Modifier.height(12.dp))

            val result = analysisResult
            if (phase >= DemoPhase.REVEALING && result != null) {
                DemoResultCard(
                    scenario = currentScenario,
                    result = result,
                    transcriptShown = transcriptShown,
                )
                Spacer(modifier = Modifier.height(12.dp))
            }

            errorMessage?.let {
                Text(text = it, color = MaterialTheme.colorScheme.error)
                Spacer(modifier = Modifier.height(8.dp))
            }

            if (phase == DemoPhase.DONE || errorMessage != null) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly,
                ) {
                    OutlinedButton(onClick = { resetScreen() }) {
                        Text("처음으로")
                    }
                }
            }
        } else {
            errorMessage?.let {
                Text(text = it, color = MaterialTheme.colorScheme.error)
                Spacer(modifier = Modifier.height(8.dp))
            }

            if (missingAudioIds.isNotEmpty()) {
                Text(
                    text = "⚠ 데모 오디오 재생 불가 (asset 누락 또는 압축): " +
                        missingAudioIds.joinToString(", ") { "#$it" },
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall,
                )
                Spacer(modifier = Modifier.height(4.dp))
            }
            if (missingTranscriptIds.isNotEmpty()) {
                Text(
                    text = "⚠ 데모 전사본 누락: " +
                        missingTranscriptIds.joinToString(", ") { "#$it" },
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall,
                )
                Spacer(modifier = Modifier.height(8.dp))
            }

            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(demoScenarios) { scenario ->
                    val status = scenarioAvailability[scenario.id]
                    val available = status?.available == true
                    val audioMissing = status?.audioPlayable == false
                    val transcriptMissing = status?.audioPlayable == true && !status.transcriptAvailable
                    DemoScenarioCard(
                        scenario = scenario,
                        enabled = phase == DemoPhase.IDLE && available,
                        missing = !available,
                        audioMissing = audioMissing,
                        transcriptMissing = transcriptMissing,
                        onAnalyze = { startScenario(scenario) },
                    )
                }
            }
        }
    }
}

@Composable
private fun DemoPlaybackSection(
    scenario: DemoScenario,
    phase: DemoPhase,
    waveform: FloatArray,
    progress: Float,
) {
    val phaseLabel = when (phase) {
        DemoPhase.PLAYING -> "🎵 재생 중 · AASIST 분석 중..."
        DemoPhase.REVEALING -> "✍ 전사 표시 중..."
        DemoPhase.DONE -> "✓ 분석 완료"
        DemoPhase.IDLE -> ""
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "#${scenario.id} ${scenario.title}",
                fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.titleMedium,
            )
            Text(
                text = phaseLabel,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(12.dp))

            WaveformView(
                waveform = waveform,
                progress = progress,
                active = phase == DemoPhase.PLAYING,
            )
            Spacer(modifier = Modifier.height(8.dp))

            val animatedProgress by animateFloatAsState(
                targetValue = progress.coerceIn(0f, 1f),
                animationSpec = tween(durationMillis = 120, easing = LinearEasing),
                label = "playback-progress",
            )
            LinearProgressIndicator(
                progress = { animatedProgress },
                modifier = Modifier.fillMaxWidth(),
            )
            if (phase == DemoPhase.PLAYING) {
                Spacer(modifier = Modifier.height(8.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(
                        modifier = Modifier.height(18.dp),
                        strokeWidth = 2.dp,
                    )
                    Spacer(modifier = Modifier.height(0.dp))
                    Text(
                        text = "  실시간 처리 중",
                        style = MaterialTheme.typography.labelSmall,
                    )
                }
            }
        }
    }
}

@Composable
private fun WaveformView(
    waveform: FloatArray,
    progress: Float,
    active: Boolean,
) {
    // 애니메이션은 active인 동안에만 작동. InfiniteTransition을 active key에 묶어
    // active=false일 때는 컴포저블이 완전히 빠지고 recomposition도 멈춘다.
    val pulseFactor = if (active) {
        val infinite = rememberInfiniteTransition(label = "waveform-pulse")
        val pulse by infinite.animateFloat(
            initialValue = 0.85f,
            targetValue = 1.15f,
            animationSpec = infiniteRepeatable(
                animation = tween(durationMillis = 600, easing = LinearEasing),
                repeatMode = RepeatMode.Reverse,
            ),
            label = "waveform-pulse",
        )
        pulse
    } else 1f

    val activeColor = MaterialTheme.colorScheme.primary
    val inactiveColor = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.45f)

    Canvas(
        modifier = Modifier
            .fillMaxWidth()
            .height(64.dp),
    ) {
        val count = waveform.size.coerceAtLeast(1)
        val gap = 2f
        val barWidth = ((size.width - gap * (count - 1)) / count).coerceAtLeast(1f)
        val centerY = size.height / 2
        val cutoff = (count * progress).toInt().coerceIn(0, count)

        for (i in 0 until count) {
            val raw = waveform[i].coerceIn(0f, 1f)
            val amplitude = (raw * pulseFactor).coerceAtMost(1f)
            val barHeight = (size.height * amplitude).coerceAtLeast(2f)
            val x = i * (barWidth + gap)
            val color = if (i < cutoff) activeColor else inactiveColor
            drawBar(x, centerY, barWidth, barHeight, color)
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawBar(
    x: Float,
    centerY: Float,
    width: Float,
    height: Float,
    color: Color,
) {
    drawRect(
        brush = SolidColor(color),
        topLeft = Offset(x, centerY - height / 2),
        size = Size(width, height),
    )
}

@Composable
private fun DemoScenarioCard(
    scenario: DemoScenario,
    enabled: Boolean,
    missing: Boolean = false,
    audioMissing: Boolean = false,
    transcriptMissing: Boolean = false,
    onAnalyze: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "#${scenario.id} ${scenario.title}",
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = scenario.description,
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    text = "예상: ${scenario.expectedResult}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary,
                )
                if (missing) {
                    val reason = when {
                        audioMissing && transcriptMissing -> "오디오 및 전사본 누락 — 시연 불가"
                        audioMissing -> "오디오 재생 불가 — 시연 불가"
                        transcriptMissing -> "전사본 누락 — 시연 불가"
                        else -> "시연 불가"
                    }
                    Text(
                        text = reason,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                }
            }
            ElevatedButton(onClick = onAnalyze, enabled = enabled) {
                Text(if (missing) "불가" else "분석")
            }
        }
    }
}

@Composable
private fun DemoResultCard(
    scenario: DemoScenario,
    result: DemoResult,
    transcriptShown: String,
) {
    val combined = result.combinedResult

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = when (combined.combinedThreatLevel) {
                CombinedThreatLevel.CRITICAL -> Color(0xFFFFCDD2)
                CombinedThreatLevel.DANGER -> Color(0xFFFFCDD2)
                CombinedThreatLevel.WARNING -> Color(0xFFFFF9C4)
                CombinedThreatLevel.CAUTION -> Color(0xFFFFF3E0)
                CombinedThreatLevel.SAFE -> Color(0xFFC8E6C9)
            },
        ),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "#${scenario.id} ${scenario.title} 분석 결과",
                fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.titleMedium,
            )
            Spacer(modifier = Modifier.height(12.dp))

            Text(
                text = "통합 위협: ${combined.combinedThreatLevel}",
                fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.titleLarge,
                color = threatColor(combined.combinedThreatLevel),
            )
            Text("예상: ${scenario.expectedResult}")
            Spacer(modifier = Modifier.height(12.dp))

            val fakeScore = result.deepfakeResult?.averageFakeScore ?: 0f
            Text("딥보이스 점수: ${(fakeScore * 100).toInt()}%")
            LinearProgressIndicator(
                progress = { fakeScore.coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth(),
                color = if (fakeScore > 0.7f) Color.Red else MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(8.dp))

            val phishingScore = result.phishingResult?.score ?: 0f
            Text("피싱 점수: ${(phishingScore * 100).toInt()}%")
            LinearProgressIndicator(
                progress = { phishingScore.coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth(),
                color = if (phishingScore > 0.3f) Color(0xFFF57F17) else MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(8.dp))

            val matched = result.phishingResult?.matchedKeywords.orEmpty()
            // 표시용 라벨은 canonical keyword, 하이라이트용은 transcript에 실제로 등장한 matchedText 포함.
            val canonicalLabels = matched.map { it.keyword }.filter { it.isNotBlank() }.distinct()
            val highlightTerms = matched
                .flatMap { listOf(it.matchedText, it.keyword) }
                .filter { it.isNotBlank() }
                .toSet()
            if (canonicalLabels.isNotEmpty()) {
                Text(
                    text = "감지 키워드: ${canonicalLabels.joinToString(", ")}",
                    color = Color.Red,
                    fontWeight = FontWeight.SemiBold,
                )
                Spacer(modifier = Modifier.height(8.dp))
            }

            if (result.transcript.isNotBlank()) {
                Text(
                    text = "전사 텍스트:",
                    fontWeight = FontWeight.SemiBold,
                    style = MaterialTheme.typography.labelMedium,
                )
                Text(
                    text = highlightKeywords(transcriptShown, highlightTerms),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }
}

private fun threatColor(level: CombinedThreatLevel): Color = when (level) {
    CombinedThreatLevel.CRITICAL -> Color.Red
    CombinedThreatLevel.DANGER -> Color(0xFFD32F2F)
    CombinedThreatLevel.WARNING -> Color(0xFFF57F17)
    CombinedThreatLevel.CAUTION -> Color(0xFFFF8F00)
    CombinedThreatLevel.SAFE -> Color(0xFF2E7D32)
}

private fun highlightKeywords(text: String, keywords: Set<String>): AnnotatedString {
    if (text.isEmpty() || keywords.isEmpty()) return AnnotatedString(text)
    val ranges = mutableListOf<IntRange>()
    // 더 긴 키워드를 먼저 매칭해 짧은 부분 매치가 긴 것을 가리지 않게 한다.
    for (kw in keywords.sortedByDescending { it.length }) {
        if (kw.isBlank()) continue
        var start = 0
        while (start <= text.length - kw.length) {
            val idx = text.indexOf(kw, start)
            if (idx < 0) break
            ranges.add(idx until idx + kw.length)
            // overlap-safe: 다음 인덱스로 한 칸만 전진 (겹치는 발생도 탐지).
            start = idx + 1
        }
    }
    if (ranges.isEmpty()) return AnnotatedString(text)
    val sorted = ranges.sortedBy { it.first }
    // 겹치는 범위를 병합 (overlap-safe 스팬 적용).
    val merged = mutableListOf<IntRange>()
    for (r in sorted) {
        val last = merged.lastOrNull()
        if (last != null && r.first <= last.last + 1) {
            merged[merged.lastIndex] = last.first..maxOf(last.last, r.last)
        } else {
            merged.add(r)
        }
    }
    return buildAnnotatedString {
        var cursor = 0
        for (r in merged) {
            if (cursor < r.first) append(text.substring(cursor, r.first))
            withStyle(SpanStyle(color = Color.Red, fontWeight = FontWeight.Bold)) {
                append(text.substring(r.first, r.last + 1))
            }
            cursor = r.last + 1
        }
        if (cursor < text.length) append(text.substring(cursor))
    }
}

private suspend fun typeTranscript(
    full: String,
    onUpdate: (String) -> Unit,
) {
    if (full.isEmpty()) {
        kotlin.coroutines.coroutineContext.ensureActive()
        onUpdate("")
        return
    }
    val chunk = 2
    var idx = 0
    while (idx < full.length) {
        // 매 chunk write 전에 활성 검사 — cancel이 delay 직후 도착했을 때 stale write 차단.
        kotlin.coroutines.coroutineContext.ensureActive()
        idx = min(idx + chunk, full.length)
        onUpdate(full.substring(0, idx))
        if (idx < full.length) delay(TYPING_INTERVAL_MS)
    }
}

/**
 * MediaPlayer로 asset WAV를 재생하고, 재생 progress(0~1)를 onTick으로 보고한다.
 * 완료되면 true, 실패하면 false를 반환한다. 호출자가 onPrepared로 MediaPlayer 참조를 받는다.
 */
private suspend fun startPlayback(
    context: android.content.Context,
    scope: CoroutineScope,
    assetPath: String,
    onPrepared: (MediaPlayer) -> Unit,
    onReleased: () -> Unit,
    onTick: (Float) -> Unit,
    attachTicker: (Job) -> Unit,
): Boolean {
    // IO 준비 단계와 Main 재생 단계 사이에 취소가 발생해도 MediaPlayer가 leak되지 않도록
    // 외부 holder에 참조를 유지하고, outer finally에서 잔여 player를 반드시 해제한다.
    val holder = AtomicReference<MediaPlayer?>(null)
    try {
        val prepared = withContext(Dispatchers.IO) {
            val afd: AssetFileDescriptor = try {
                context.assets.openFd(assetPath)
            } catch (_: Exception) {
                return@withContext null
            }
            val mp = MediaPlayer()
            var ownershipTransferred = false
            try {
                mp.setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                mp.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
                mp.prepare()
                val duration = mp.duration.coerceAtLeast(1)
                holder.set(mp) // outer finally가 볼 수 있도록 먼저 등록.
                ownershipTransferred = true
                mp to duration
            } catch (ce: kotlinx.coroutines.CancellationException) {
                throw ce
            } catch (_: Exception) {
                null
            } finally {
                runCatching { afd.close() }
                if (!ownershipTransferred) runCatching { mp.release() }
            }
        } ?: return false

        val (mp, durationMs) = prepared

        return withContext(Dispatchers.Main) {
            onPrepared(mp)

            val completion = CompletableDeferred<Boolean>()
            mp.setOnCompletionListener {
                if (!completion.isCompleted) completion.complete(true)
            }
            mp.setOnErrorListener { _, _, _ ->
                if (!completion.isCompleted) completion.complete(false)
                true
            }

            try {
                mp.start()
            } catch (_: Exception) {
                return@withContext false
            }

            val ticker: Job = scope.launch(Dispatchers.Main) {
                try {
                    while (true) {
                        ensureActive()
                        val pos = runCatching { mp.currentPosition }.getOrDefault(0)
                        val ratio = (pos.toFloat() / durationMs).coerceIn(0f, 1f)
                        onTick(ratio)
                        if (ratio >= 0.999f) break
                        delay(PLAYBACK_TICK_MS)
                    }
                } catch (_: Throwable) {
                    // completion listener가 종료를 마무리한다.
                }
            }
            attachTicker(ticker)

            try {
                val ok = completion.await()
                ticker.cancel()
                ensureActive()
                // 성공 시에만 progress bar를 100%로 완결. 실패/에러는 진행률을 건드리지 않는다.
                if (ok) onTick(1f)
                ok
            } finally {
                ticker.cancel()
            }
        }
    } finally {
        // IO/Main 사이 취소, Main 내부 취소/에러 등 모든 경로에서 단일 지점에서 해제.
        val leaked = holder.getAndSet(null)
        if (leaked != null) {
            runCatching { if (leaked.isPlaying) leaked.stop() }
            runCatching { leaked.release() }
            onReleased()
        }
    }
}

/**
 * WAV 파일에서 "data" 서브청크의 시작 오프셋과 크기를 찾는다.
 * RIFF 구조: "RIFF" + size(4) + "WAVE" + 반복(subchunkId(4) + subchunkSize(4) + data...).
 * demo WAV는 "LIST"(metadata) 등 optional chunk를 포함할 수 있어 고정 44바이트 오프셋은 잘못됨.
 */
private fun findWavDataChunk(bytes: ByteArray): Pair<Int, Int>? {
    if (bytes.size < 12) return null
    // "RIFF" + size + "WAVE" 검증
    val riff = String(bytes, 0, 4, Charsets.US_ASCII)
    val wave = String(bytes, 8, 4, Charsets.US_ASCII)
    if (riff != "RIFF" || wave != "WAVE") return null
    var offset = 12
    while (offset + 8 <= bytes.size) {
        val id = String(bytes, offset, 4, Charsets.US_ASCII)
        val size = ByteBuffer.wrap(bytes, offset + 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
        if (size < 0) return null
        val dataStart = offset + 8
        if (id == "data") {
            val actual = minOf(size, bytes.size - dataStart).coerceAtLeast(0)
            return dataStart to actual
        }
        offset = dataStart + size + (size and 1) // word-align
    }
    return null
}

/** WAV asset에서 16-bit PCM 샘플을 파싱해 WAVEFORM_BUCKETS 크기 진폭 배열로 다운샘플링한다. */
private fun loadWaveform(context: android.content.Context, assetPath: String): FloatArray {
    val buckets = WAVEFORM_BUCKETS
    return try {
        val bytes = context.assets.open(assetPath).use { it.readBytes() }
        val chunk = findWavDataChunk(bytes) ?: return FloatArray(buckets)
        val dataStart = chunk.first
        val dataSize = chunk.second
        val numSamples = dataSize / 2
        if (numSamples <= 0) return FloatArray(buckets)
        val samplesPerBucket = (numSamples / buckets).coerceAtLeast(1)
        val buffer = ByteBuffer.wrap(bytes, dataStart, dataSize).order(ByteOrder.LITTLE_ENDIAN)
        val result = FloatArray(buckets)
        var peak = 0f
        for (b in 0 until buckets) {
            var maxAbs = 0f
            val end = min(samplesPerBucket, buffer.remaining() / 2)
            for (s in 0 until end) {
                val sample = buffer.short.toInt()
                // Short.MIN_VALUE(-32768)는 Kotlin abs()가 음수를 반환하므로 직접 clamp.
                val absValue = if (sample == Short.MIN_VALUE.toInt()) 32768 else abs(sample)
                val normalized = absValue / 32768f
                if (normalized > maxAbs) maxAbs = normalized
            }
            result[b] = maxAbs
            if (maxAbs > peak) peak = maxAbs
        }
        // 피크가 매우 작으면 시각성 확보를 위해 스케일 업 (무음 방지).
        if (peak in 0.001f..0.25f) {
            val scale = 0.6f / peak
            for (i in result.indices) result[i] = (result[i] * scale).coerceAtMost(1f)
        }
        result
    } catch (_: Exception) {
        FloatArray(buckets)
    }
}

