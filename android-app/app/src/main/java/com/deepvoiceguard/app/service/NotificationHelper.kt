package com.deepvoiceguard.app.service

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.deepvoiceguard.app.R
import com.deepvoiceguard.app.inference.AggregatedResult
import com.deepvoiceguard.app.inference.CombinedThreatLevel
import com.deepvoiceguard.app.inference.CombinedThreatResult
import com.deepvoiceguard.app.inference.ThreatLevel
import com.deepvoiceguard.app.ui.MainActivity
import java.util.concurrent.atomic.AtomicLong

/**
 * Deepfake + 피싱 탐지 알림을 생성하고 표시한다.
 * - 쿨다운: 같은 위협 레벨 알림은 최소 30초 간격
 * - 잠금화면: VISIBILITY_PRIVATE — 키워드/전사 미노출
 * - 피싱 알림: 최종 전사 확정 후에만 발송
 */
class NotificationHelper(private val context: Context) {

    companion object {
        const val ALERT_CHANNEL_ID = "deepvoice_alert_channel"
        const val PHISHING_CHANNEL_ID = "phishing_alert_channel"
        const val COMBINED_CHANNEL_ID = "combined_alert_channel"
        private const val SERVICE_ERROR_NOTIFICATION_ID = 2001
        // long-running foreground service에서 1001회 이상 post 시 SERVICE_ERROR_NOTIFICATION_ID(2001)과
        // 충돌해 '탐지 중단' 배너를 덮어쓰거나 그 반대로 서로를 지우는 문제가 있어 10_000부터 시작한다.
        private var notificationId = 10_000
        private const val COOLDOWN_MS = 30_000L
        // **H106**: incident는 이 시간 동안 elevated alert가 없으면 만료되어 rank가 리셋된다.
        // 만료 이후 같은 session에서 새 WARNING/CAUTION 이벤트가 생겨도 post 가능.
        private const val INCIDENT_EXPIRY_MS = 90_000L
    }

    // 쿨다운 state — cross-thread 접근 방어용 synchronization.
    private val stateLock = Any()
    // 쿨다운: 위협 레벨별 마지막 알림 시간
    private val lastAlertTime = mutableMapOf<String, Long>()

    // Incident별 마지막으로 post된 severity rank — escalation bypass 판단용.
    // key: "session|<sessionId>", value: severity rank (0=SAFE, 1=CAUTION, 2=WARNING, 3=DANGER, 4=CRITICAL).
    private val lastIncidentRank = mutableMapOf<String, Int>()

    // **H106/H108**: session별 마지막으로 elevated threat가 *관찰*된 시각. post/suppress 여부와
    // 무관하게 showCombinedAlert/showDetectionAlert 호출 시마다 갱신한다. INCIDENT_EXPIRY_MS
    // 초과 후에만 rank가 자동 리셋되어 새 incident window가 시작된다. 연속된 WARNING/CAUTION이
    // expiry에 걸려 새 incident로 재오픈되는 버그 차단.
    private val lastIncidentActivityTime = mutableMapOf<String, Long>()

    // **H115/H117/H118**: retry와 cooldown 모두 slot(session+level+channel) 단위로 추적한다.
    // WARNING 경로가 phishing↔deepfake 채널을 오갈 수 있으므로 각 channel을 독립 delivery/
    // cooldown 단위로 기록한다.
    private val deliveredSlots = mutableSetOf<String>()

    // **H121**: rollback ownership을 위한 unique attempt counter. wall-clock 충돌(같은 ms 두
    // 이벤트) 시 ms 기반 가드가 newer state를 stale state로 덮어쓰는 race를 차단.
    private val attemptCounter = AtomicLong(0L)
    private val cooldownAttemptOwner = mutableMapOf<String, Long>()
    private val rankAttemptOwner = mutableMapOf<String, Long>()

    /**
     * **H162**: session별 마지막으로 발송된 notification id. SAFE 전환 시 NotificationManager로
     * 명시 cancel하여 stale CRITICAL 알림이 트레이에 남는 것을 방지.
     */
    private val postedNotificationIds = mutableMapOf<String, MutableList<Int>>()

    /** Test seam: ownership check 직전에 concurrent overwrite를 재현한다. */
    internal var beforeOwnershipRecheck: (() -> Unit)? = null

    private fun sessionKey(callSessionId: String?): String = "session|${callSessionId ?: "no-session"}"

    /**
     * 모니터링 세션이 종료될 때 호출하여 해당 session의 쿨다운/escalation 상태를 제거한다.
     * null이면 모든 상태를 초기화한다 (전체 shutdown 경로).
     */
    fun clearIncidentState(sessionId: String?) {
        val toCancel = synchronized(stateLock) {
            if (sessionId == null) {
                lastIncidentRank.clear()
                lastAlertTime.clear()
                lastIncidentActivityTime.clear()
                deliveredSlots.clear()
                cooldownAttemptOwner.clear()
                rankAttemptOwner.clear()
                val all = postedNotificationIds.values.flatten().toList()
                postedNotificationIds.clear()
                all
            } else {
                val prefix = "session|$sessionId"
                lastIncidentRank.remove(prefix)
                lastIncidentActivityTime.remove(prefix)
                rankAttemptOwner.remove(prefix)
                val incidentPrefix = "incident|${sessionId}|"
                lastAlertTime.keys.toList().filter { it.startsWith(incidentPrefix) }
                    .forEach { lastAlertTime.remove(it); cooldownAttemptOwner.remove(it) }
                deliveredSlots.toList().filter { it.startsWith(incidentPrefix) }
                    .forEach { deliveredSlots.remove(it) }
                postedNotificationIds.remove(prefix) ?: emptyList()
            }
        }
        val nm = NotificationManagerCompat.from(context)
        for (id in toCancel) runCatching { nm.cancel(id) }
    }

    /**
     * **H120**: incident 만료 시 호출. 해당 sessionPart의 deliveredSlots/cooldown 상태를 정리.
     * stateLock을 보유한 상태에서 호출해야 한다.
     */
    private fun resetIncidentSlotsLocked(sessionPart: String) {
        val incidentPrefix = "incident|${sessionPart}|"
        deliveredSlots.toList().filter { it.startsWith(incidentPrefix) }
            .forEach { deliveredSlots.remove(it) }
        // cooldown/Owner도 함께 정리 — 새 incident가 깨끗한 상태로 시작.
        lastAlertTime.keys.toList().filter { it.startsWith(incidentPrefix) }
            .forEach { lastAlertTime.remove(it); cooldownAttemptOwner.remove(it) }
    }

    private fun threatLevelRank(level: ThreatLevel): Int = when (level) {
        ThreatLevel.SAFE -> 0
        ThreatLevel.CAUTION -> 1
        ThreatLevel.WARNING -> 2
        ThreatLevel.DANGER -> 3
    }

    private fun combinedLevelRank(level: CombinedThreatLevel): Int = when (level) {
        CombinedThreatLevel.SAFE -> 0
        CombinedThreatLevel.CAUTION -> 1
        CombinedThreatLevel.WARNING -> 2
        CombinedThreatLevel.DANGER -> 3
        CombinedThreatLevel.CRITICAL -> 4
    }

    init {
        createChannels()
    }

    private fun createChannels() {
        val nm = context.getSystemService(NotificationManager::class.java)

        // 딥보이스 알림 채널
        nm.createNotificationChannel(
            NotificationChannel(
                ALERT_CHANNEL_ID, "딥보이스 알림", NotificationManager.IMPORTANCE_HIGH,
            ).apply {
                description = "AI 합성 음성 탐지 알림"
                enableVibration(true)
                vibrationPattern = longArrayOf(0, 500, 200, 500)
            }
        )

        // 피싱 알림 채널
        nm.createNotificationChannel(
            NotificationChannel(
                PHISHING_CHANNEL_ID, "보이스피싱 알림", NotificationManager.IMPORTANCE_HIGH,
            ).apply {
                description = "보이스피싱 키워드 탐지 알림"
                enableVibration(true)
                vibrationPattern = longArrayOf(0, 400, 200, 400)
            }
        )

        // 통합 CRITICAL 알림 채널
        nm.createNotificationChannel(
            NotificationChannel(
                COMBINED_CHANNEL_ID, "긴급 위협 알림", NotificationManager.IMPORTANCE_HIGH,
            ).apply {
                description = "딥보이스 + 보이스피싱 동시 탐지"
                enableVibration(true)
                vibrationPattern = longArrayOf(0, 700, 300, 700)
            }
        )
    }

    /**
     * **H124**: combined 결과가 SAFE로 떨어진 시점에 호출. 다음 elevated event를 fresh
     * incident로 인식하도록 해당 session의 incident state를 정리한다.
     * lastIncidentActivityTime은 그대로 두지 않고 제거하여 expiry 로직 trigger 시 깔끔하게 시작.
     */
    fun onIncidentSafe(callSessionId: String?) {
        val toCancel = synchronized(stateLock) {
            val sessionStr = sessionKey(callSessionId)
            val sessionPart = callSessionId ?: "no-session"
            lastIncidentRank.remove(sessionStr)
            rankAttemptOwner.remove(sessionStr)
            lastIncidentActivityTime.remove(sessionStr)
            resetIncidentSlotsLocked(sessionPart)
            // **H162**: 발송된 notification id snapshot 후 lock 외부에서 cancel.
            postedNotificationIds.remove(sessionStr) ?: emptyList()
        }
        // NotificationManagerCompat.cancel은 lock 밖에서 실행 (system call).
        val nm = NotificationManagerCompat.from(context)
        for (id in toCancel) runCatching { nm.cancel(id) }
    }

    /**
     * **H109**: post/suppress와 무관하게 incident 관찰 시점을 갱신한다.
     * suppressAlert=true 인 persistence-only fallback 경로에서도 호출하여 연속 incident의
     * 90s 만료로 인한 rank 리셋을 막는다.
     */
    fun touchIncidentActivity(callSessionId: String?) = synchronized(stateLock) {
        val sessionStr = sessionKey(callSessionId)
        lastIncidentActivityTime[sessionStr] = System.currentTimeMillis()
    }

    /**
     * **H163/H165**: severity downgrade — stale 트레이 알림 cancel + replacement alert force post.
     * 단순 cancel만 하면 incident 활성 상태에서도 트레이가 비어 사용자가 경계 해제로 착각한다.
     * 따라서 cooldown/rank/delivered slot을 새 level 기준으로 정리하고 즉시 새 alert를 post한다.
     */
    fun onIncidentDowngrade(callSessionId: String?, replacement: CombinedThreatResult?) {
        val sessionStr = sessionKey(callSessionId)
        val sessionPart = callSessionId ?: "no-session"
        val replacementValid = replacement?.takeIf {
            it.combinedThreatLevel != CombinedThreatLevel.SAFE
        }
        if (replacementValid != null) {
            // **H170/H172/H177/H179**: oldIds snapshot부터 cancel 결정까지 단일 atomic transaction.
            // 외부에서 oldIds 잡으면 lock gap 동안 concurrent post nid가 추가돼 stale CRITICAL이
            // 트레이에 살아남음. 모든 mutation을 reentrant synchronized 안에서 수행.
            val incidentPrefix = "incident|${sessionPart}|"
            data class DowngradeSnapshot(
                val rank: Int?,
                val alertTimes: Map<String, Long>,
                val delivered: Set<String>,
                val activityTime: Long?,
            )
            val toCancel = synchronized(stateLock) {
                val oldIds = (postedNotificationIds[sessionStr] ?: emptyList()).toList()
                val snapshot = DowngradeSnapshot(
                    rank = lastIncidentRank[sessionStr],
                    alertTimes = lastAlertTime.filter { it.key.startsWith(incidentPrefix) }.toMap(),
                    delivered = deliveredSlots.filter { it.startsWith(incidentPrefix) }.toSet(),
                    activityTime = lastIncidentActivityTime[sessionStr],
                )
                lastIncidentRank[sessionStr] = combinedLevelRank(replacementValid.combinedThreatLevel)
                resetIncidentSlotsLocked(sessionPart)
                val posted = showCombinedAlert(replacementValid, callSessionId)
                if (posted) {
                    // commit — oldIds를 postedNotificationIds에서 제거 후 caller에게 cancel 위임.
                    postedNotificationIds[sessionStr]?.removeAll(oldIds.toSet())
                    oldIds
                } else {
                    // rollback — snapshot 복원, cancel 안 함.
                    snapshot.rank?.let { lastIncidentRank[sessionStr] = it }
                        ?: lastIncidentRank.remove(sessionStr)
                    snapshot.alertTimes.forEach { lastAlertTime[it.key] = it.value }
                    deliveredSlots.addAll(snapshot.delivered)
                    snapshot.activityTime?.let { lastIncidentActivityTime[sessionStr] = it }
                        ?: lastIncidentActivityTime.remove(sessionStr)
                    emptyList()
                }
            }
            if (toCancel.isNotEmpty()) {
                val nm = NotificationManagerCompat.from(context)
                for (id in toCancel) runCatching { nm.cancel(id) }
            }
        } else {
            // replacement이 없거나 SAFE — 그냥 cancel (SAFE는 onIncidentSafe 권장).
            val toCancel = synchronized(stateLock) {
                lastIncidentRank.remove(sessionStr)
                resetIncidentSlotsLocked(sessionPart)
                postedNotificationIds.remove(sessionStr) ?: emptyList()
            }
            val nm = NotificationManagerCompat.from(context)
            for (id in toCancel) runCatching { nm.cancel(id) }
        }
    }

    /**
     * **H111**: suppressAlert fallback 경로에서 사용. 첫 elevated 이벤트가 권한 거부로 post되지
     * 못했을 때, 이후 stable incident에 대한 fallback 이벤트에서 alert를 재시도한다.
     * 이미 post된 적이 있으면 activity만 갱신하고 post는 생략.
     */
    fun ensureIncidentAlerted(
        result: CombinedThreatResult,
        callSessionId: String?,
    ) {
        // **H112/H115/H117**: retry 결정을 현재 alert *slot* (level+channel) 기준으로 판단.
        // lastAlertTime[slotKey]==0이면 이 slot이 아직 한 번도 성공적으로 posted되지 않은 것.
        // lastIncidentRank는 session 전체를 모으므로 같은 rank의 다른 channel(phishing vs
        // deepfake WARNING)이 부분 실패해도 재시도가 정상 동작하지 않는 문제가 있었다.
        val routed = routeCombinedAlert(result) ?: return
        val sessionPart = callSessionId ?: "no-session"
        val slotKey = deliverySlotKey(sessionPart, result.combinedThreatLevel, routed.channelId)
        val shouldRetry = synchronized(stateLock) {
            val sessionStr = sessionKey(callSessionId)
            val now = System.currentTimeMillis()
            val lastActivity = lastIncidentActivityTime[sessionStr] ?: 0L
            if (lastActivity > 0 && (now - lastActivity) > INCIDENT_EXPIRY_MS) {
                lastIncidentRank.remove(sessionStr)
                rankAttemptOwner.remove(sessionStr)
                // **H120**: incident 만료 시 delivered slot도 정리 — 새 incident가 같은
                // slot에서 retry 가능.
                resetIncidentSlotsLocked(sessionPart)
            }
            lastIncidentActivityTime[sessionStr] = now
            slotKey !in deliveredSlots
        }
        if (shouldRetry) {
            showCombinedAlert(result, callSessionId)
        }
    }

    private data class RoutedCombinedAlert(
        val channelId: String,
        val title: String,
        val text: String,
    )

    /**
     * CombinedThreatResult를 channel/title/text로 매핑. cooldown key와 실제 post에서 동일한
     * 채널이 선택되도록 helper로 추출 (H117).
     */
    private fun routeCombinedAlert(result: CombinedThreatResult): RoutedCombinedAlert? {
        val keywordSummary = if (result.matchedKeywords.isNotEmpty()) {
            result.matchedKeywords.take(3).joinToString(", ") { it.keyword }
        } else null
        return when (result.combinedThreatLevel) {
            CombinedThreatLevel.CRITICAL -> RoutedCombinedAlert(
                COMBINED_CHANNEL_ID,
                "긴급: 딥보이스 + 보이스피싱 감지!",
                "AI 합성 음성과 피싱 패턴이 동시에 감지되었습니다" +
                    (if (keywordSummary != null) " [$keywordSummary]" else ""),
            )
            CombinedThreatLevel.DANGER -> RoutedCombinedAlert(
                ALERT_CHANNEL_ID,
                "딥보이스 감지!",
                "AI 합성 음성이 감지되었습니다",
            )
            CombinedThreatLevel.WARNING -> {
                if (result.phishingScore > 0.3f) {
                    RoutedCombinedAlert(
                        PHISHING_CHANNEL_ID,
                        "보이스피싱 의심",
                        "피싱 패턴이 감지되었습니다. 주의하세요." +
                            (if (keywordSummary != null) " [$keywordSummary]" else ""),
                    )
                } else {
                    RoutedCombinedAlert(ALERT_CHANNEL_ID, "의심스러운 음성", "주의가 필요합니다")
                }
            }
            CombinedThreatLevel.CAUTION -> RoutedCombinedAlert(
                ALERT_CHANNEL_ID,
                "주의 필요",
                "약간의 위험 징후가 감지되었습니다",
            )
            CombinedThreatLevel.SAFE -> null
        }
    }

    /**
     * cooldown은 session + severity + channel로 keyed.
     * WARNING 경로가 ALERT/PHISHING 채널을 오갈 수 있으므로 채널별로 독립 cooldown slot을 유지한다.
     */
    private fun cooldownKey(sessionPart: String, level: CombinedThreatLevel, channelId: String) =
        "incident|$sessionPart|$level|$channelId"

    private fun cooldownKeyDeepfake(sessionPart: String, level: ThreatLevel, channelId: String) =
        "incident|$sessionPart|$level|$channelId"

    private fun deliverySlotKey(
        sessionPart: String,
        level: CombinedThreatLevel,
        channelId: String,
    ) = "incident|$sessionPart|$level|$channelId"

    private fun deliverySlotKeyDeepfake(
        sessionPart: String,
        level: ThreatLevel,
        channelId: String,
    ) = "incident|$sessionPart|$level|$channelId"

    /** 기존 딥보이스 전용 알림. */
    fun showDetectionAlert(result: AggregatedResult, callSessionId: String? = null): Boolean {
        if (result.threatLevel == ThreatLevel.SAFE) return false
        val sessionPart = callSessionId ?: "no-session"
        val newRank = threatLevelRank(result.threatLevel)
        val channelId = ALERT_CHANNEL_ID
        val cKey = cooldownKeyDeepfake(sessionPart, result.threatLevel, channelId)
        val dKey = deliverySlotKeyDeepfake(sessionPart, result.threatLevel, channelId)
        val sessionStr = sessionKey(callSessionId)
        var prevRankCaptured = -1
        var prevAlertTime = 0L
        var rankWasIncremented = false
        // **H121**: unique attempt token — wall-clock 충돌 가능성 제거.
        val myAttempt = attemptCounter.incrementAndGet()
        val shouldPost = synchronized(stateLock) {
            val now = System.currentTimeMillis()
            val lastActivity = lastIncidentActivityTime[sessionStr] ?: 0L
            if (lastActivity > 0 && (now - lastActivity) > INCIDENT_EXPIRY_MS) {
                lastIncidentRank.remove(sessionStr)
                rankAttemptOwner.remove(sessionStr)
                resetIncidentSlotsLocked(sessionPart)
            }
            lastIncidentActivityTime[sessionStr] = now
            val prevRank = lastIncidentRank[sessionStr] ?: -1
            prevRankCaptured = prevRank
            prevAlertTime = lastAlertTime[cKey] ?: 0L
            val isEscalation = newRank > prevRank
            val cooldownActive = (now - prevAlertTime) < COOLDOWN_MS
            val willPost = when {
                !isEscalation && cooldownActive -> false
                newRank < prevRank -> false
                else -> true
            }
            if (willPost && canPostNotification()) {
                lastAlertTime[cKey] = now
                cooldownAttemptOwner[cKey] = myAttempt
                if (newRank > prevRank) {
                    lastIncidentRank[sessionStr] = newRank
                    rankAttemptOwner[sessionStr] = myAttempt
                    rankWasIncremented = true
                }
                true
            } else {
                false
            }
        }
        if (!shouldPost) return false

        val (title, text) = when (result.threatLevel) {
            ThreatLevel.DANGER -> "딥보이스 감지!" to
                    "AI 합성 음성이 감지되었습니다 (${(result.averageFakeScore * 100).toInt()}%)"
            ThreatLevel.WARNING -> "의심스러운 음성" to
                    "연속 의심 구간 감지 (${(result.averageFakeScore * 100).toInt()}%)"
            ThreatLevel.CAUTION -> "주의 필요" to
                    "약간 높은 딥보이스 점수 (${(result.averageFakeScore * 100).toInt()}%)"
            else -> return false
        }

        // **H180**: postNotification 호출을 synchronized 안에서 + ownership 재검.
        // onIncidentDowngrade 등이 cooldownAttemptOwner를 invalidate한 경우 stale poster는 여기서
        // 차단되어 트레이에 stale 알림 잔존하지 않음.
        beforeOwnershipRecheck?.invoke()
        val posted = synchronized(stateLock) {
            if (cooldownAttemptOwner[cKey] != myAttempt) {
                return@synchronized false  // ownership lost — onIncidentDowngrade가 reset.
            }
            val ok = postNotification(channelId, title, text, result.threatLevel.toPriority(), sessionStr)
            if (ok) {
                deliveredSlots.add(dKey)
            } else {
                cooldownAttemptOwner.remove(cKey)
                if (prevAlertTime == 0L) lastAlertTime.remove(cKey)
                else lastAlertTime[cKey] = prevAlertTime
                if (rankWasIncremented && rankAttemptOwner[sessionStr] == myAttempt) {
                    rankAttemptOwner.remove(sessionStr)
                    if (prevRankCaptured < 0) lastIncidentRank.remove(sessionStr)
                    else lastIncidentRank[sessionStr] = prevRankCaptured
                }
            }
            ok
        }
        return posted
    }

    /** 통합 위협 알림 (딥보이스 + 피싱). */
    fun showCombinedAlert(result: CombinedThreatResult, callSessionId: String? = null): Boolean {
        if (result.combinedThreatLevel == CombinedThreatLevel.SAFE) return false
        val sessionPart = callSessionId ?: "no-session"
        val newRank = combinedLevelRank(result.combinedThreatLevel)
        val sessionStr = sessionKey(callSessionId)

        val routed = routeCombinedAlert(result) ?: return false
        val channelId = routed.channelId
        val title = routed.title
        val text = routed.text
        val cKey = cooldownKey(sessionPart, result.combinedThreatLevel, channelId)
        val dKey = deliverySlotKey(sessionPart, result.combinedThreatLevel, channelId)
        var prevRankCaptured = -1
        var prevAlertTime = 0L
        var rankWasIncremented = false
        val myAttempt = attemptCounter.incrementAndGet()
        val shouldPost = synchronized(stateLock) {
            val now = System.currentTimeMillis()
            val lastActivity = lastIncidentActivityTime[sessionStr] ?: 0L
            if (lastActivity > 0 && (now - lastActivity) > INCIDENT_EXPIRY_MS) {
                lastIncidentRank.remove(sessionStr)
                rankAttemptOwner.remove(sessionStr)
                resetIncidentSlotsLocked(sessionPart)
            }
            lastIncidentActivityTime[sessionStr] = now
            val prevRank = lastIncidentRank[sessionStr] ?: -1
            prevRankCaptured = prevRank
            prevAlertTime = lastAlertTime[cKey] ?: 0L
            val isEscalation = newRank > prevRank
            val cooldownActive = (now - prevAlertTime) < COOLDOWN_MS
            val willPost = when {
                !isEscalation && cooldownActive -> false
                newRank < prevRank -> false
                else -> true
            }
            if (willPost && canPostNotification()) {
                lastAlertTime[cKey] = now
                cooldownAttemptOwner[cKey] = myAttempt
                if (newRank > prevRank) {
                    lastIncidentRank[sessionStr] = newRank
                    rankAttemptOwner[sessionStr] = myAttempt
                    rankWasIncremented = true
                }
                true
            } else {
                false
            }
        }
        if (!shouldPost) return false

        val priority = when (result.combinedThreatLevel) {
            CombinedThreatLevel.CRITICAL -> NotificationCompat.PRIORITY_MAX
            CombinedThreatLevel.DANGER -> NotificationCompat.PRIORITY_HIGH
            CombinedThreatLevel.WARNING -> NotificationCompat.PRIORITY_HIGH
            else -> NotificationCompat.PRIORITY_DEFAULT
        }

        // **H180**: post를 synchronized 안에서 + ownership 재검.
        beforeOwnershipRecheck?.invoke()
        val posted = synchronized(stateLock) {
            if (cooldownAttemptOwner[cKey] != myAttempt) {
                return@synchronized false
            }
            val ok = postNotification(channelId, title, text, priority, sessionStr)
            if (ok) {
                deliveredSlots.add(dKey)
            } else {
                cooldownAttemptOwner.remove(cKey)
                if (prevAlertTime == 0L) lastAlertTime.remove(cKey)
                else lastAlertTime[cKey] = prevAlertTime
                if (rankWasIncremented && rankAttemptOwner[sessionStr] == myAttempt) {
                    rankAttemptOwner.remove(sessionStr)
                    if (prevRankCaptured < 0) lastIncidentRank.remove(sessionStr)
                    else lastIncidentRank[sessionStr] = prevRankCaptured
                }
            }
            ok
        }
        return posted
    }

    /**
     * 서비스 장애를 사용자에게 즉시 surface하는 persistent 오류 알림.
     * Foreground service 알림과 별개로 유지되어 capture 중단을 명확히 드러낸다.
     */
    fun showServiceError(message: String): Boolean {
        if (!isChannelEnabled(ALERT_CHANNEL_ID)) {
            android.util.Log.w("NotificationHelper", "channel $ALERT_CHANNEL_ID disabled — treating service error post as failed")
            return false
        }
        if (!canPostNotification()) {
            android.util.Log.w("NotificationHelper", "notifications disabled — treating service error post as failed")
            return false
        }
        return try {
            val intent = Intent(context, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
            }
            val pendingIntent = PendingIntent.getActivity(
                context, 0, intent, PendingIntent.FLAG_IMMUTABLE,
            )
            val normalizedMessage = message
                .lineSequence()
                .firstOrNull()
                ?.trim()
                ?.take(120)
                .orEmpty()
                .ifBlank { "오류를 확인해 주세요" }
            val notification = NotificationCompat.Builder(context, ALERT_CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_shield)
                .setContentTitle("탐지가 중단되었습니다")
                .setContentText(normalizedMessage)
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setContentIntent(pendingIntent)
                .setAutoCancel(false)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .setVisibility(NotificationCompat.VISIBILITY_PRIVATE)
                .build()

            NotificationManagerCompat.from(context).notify(SERVICE_ERROR_NOTIFICATION_ID, notification)
            true
        } catch (t: Throwable) {
            android.util.Log.w("NotificationHelper", "showServiceError failed", t)
            false
        }
    }

    /**
     * 실제 notify() 호출. **H113**: 성공/실패 Boolean 반환하여 caller가 rollback 할 수 있게.
     */
    private fun postNotification(
        channelId: String,
        title: String,
        text: String,
        priority: Int,
        sessionStr: String? = null,
    ): Boolean {
        // **H114**: 채널이 비활성이면 notify는 no-op이지만 throw하지 않는다 → 명시 실패 반환.
        if (!isChannelEnabled(channelId)) {
            android.util.Log.w(
                "NotificationHelper",
                "channel $channelId disabled — treating post as failed",
            )
            return false
        }
        return try {
            val intent = Intent(context, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
            }
            val pendingIntent = PendingIntent.getActivity(
                context, 0, intent, PendingIntent.FLAG_IMMUTABLE,
            )

            val notification = NotificationCompat.Builder(context, channelId)
                .setSmallIcon(R.drawable.ic_shield)
                .setContentTitle(title)
                .setContentText(text)
                .setPriority(priority)
                .setContentIntent(pendingIntent)
                .setAutoCancel(true)
                .setVisibility(NotificationCompat.VISIBILITY_PRIVATE)
                .build()

            val nid = notificationId++
            NotificationManagerCompat.from(context).notify(nid, notification)
            // **H162**: session별 notification id 기록 — onIncidentSafe에서 cancel 가능.
            sessionStr?.let { sid ->
                synchronized(stateLock) {
                    postedNotificationIds.getOrPut(sid) { mutableListOf() }.add(nid)
                }
            }
            true
        } catch (t: Throwable) {
            android.util.Log.w("NotificationHelper", "postNotification failed", t)
            false
        }
    }

    /**
     * Channel-level visibility 확인 — channel importance가 IMPORTANCE_NONE이면 block 상태.
     * API 26+ notification channel을 쓰므로 반드시 채널별 차단 여부를 별도 체크해야 한다.
     */
    private fun isChannelEnabled(channelId: String): Boolean {
        val nm = context.getSystemService(NotificationManager::class.java) ?: return true
        val channel = nm.getNotificationChannel(channelId) ?: return true
        return channel.importance != NotificationManager.IMPORTANCE_NONE
    }

    private fun canPostNotification(): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(
                    context, Manifest.permission.POST_NOTIFICATIONS,
                ) != PackageManager.PERMISSION_GRANTED
            ) return false
        }
        // **H114**: 앱 레벨 notification 토글 — 사용자가 시스템 설정에서 꺼놨으면 notify가
        // 조용히 no-op 되지만, 우리는 그 상태를 "실패"로 간주해 incident 상태가 잘못 posted로
        // 기록되지 않게 해야 한다.
        return NotificationManagerCompat.from(context).areNotificationsEnabled()
    }

    private fun ThreatLevel.toPriority(): Int = when (this) {
        ThreatLevel.DANGER -> NotificationCompat.PRIORITY_MAX
        ThreatLevel.WARNING -> NotificationCompat.PRIORITY_HIGH
        else -> NotificationCompat.PRIORITY_DEFAULT
    }
}
