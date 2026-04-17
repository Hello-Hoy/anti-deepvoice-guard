package com.deepvoiceguard.app

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.deepvoiceguard.app.inference.AggregatedResult
import com.deepvoiceguard.app.inference.CombinedThreatLevel
import com.deepvoiceguard.app.inference.CombinedThreatResult
import com.deepvoiceguard.app.inference.DetectionResult
import com.deepvoiceguard.app.inference.ThreatLevel
import com.deepvoiceguard.app.phishing.model.PhishingResult
import com.deepvoiceguard.app.phishing.model.PhishingThreatLevel
import com.deepvoiceguard.app.service.NotificationHelper
import com.deepvoiceguard.app.stt.SttStatus
import io.mockk.Runs
import io.mockk.every
import io.mockk.just
import io.mockk.mockk
import io.mockk.mockkStatic
import io.mockk.spyk
import io.mockk.unmockkAll
import io.mockk.verify
import java.lang.reflect.Field
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class NotificationHelperTest {

    private lateinit var context: Context
    private lateinit var notificationManager: NotificationManager
    private lateinit var notificationManagerCompat: NotificationManagerCompat
    private lateinit var helper: NotificationHelper

    @Before
    fun setUp() {
        context = mockk()
        notificationManager = mockk(relaxed = true)
        notificationManagerCompat = mockk(relaxed = true)

        mockkStatic(NotificationManagerCompat::class)
        mockkStatic(ContextCompat::class)

        every { context.getSystemService(NotificationManager::class.java) } returns notificationManager
        every { notificationManager.createNotificationChannel(any()) } just Runs
        every { notificationManager.getNotificationChannel(any()) } returns null
        every { NotificationManagerCompat.from(context) } returns notificationManagerCompat
        every { notificationManagerCompat.areNotificationsEnabled() } returns true
        every { notificationManagerCompat.notify(any<Int>(), any()) } just Runs
        every { notificationManagerCompat.cancel(any<Int>()) } just Runs
        every { ContextCompat.checkSelfPermission(any(), any()) } returns android.content.pm.PackageManager.PERMISSION_GRANTED

        helper = spyk(NotificationHelper(context), recordPrivateCalls = true)
    }

    @After
    fun tearDown() {
        unmockkAll()
    }

    @Test
    fun `H104 detection alerts never post on rank downgrade inside same incident`() {
        stubPostNotification(result = true)

        assertTrue(helper.showDetectionAlert(detectionResult(ThreatLevel.DANGER), SESSION_ID))
        assertFalse(helper.showDetectionAlert(detectionResult(ThreatLevel.WARNING), SESSION_ID))
        assertFalse(helper.showDetectionAlert(detectionResult(ThreatLevel.CAUTION), SESSION_ID))
    }

    @Test
    fun `H104 combined alerts never post on rank downgrade inside same incident`() {
        stubPostNotification(result = true)

        assertTrue(helper.showCombinedAlert(combinedResult(CombinedThreatLevel.CRITICAL, phishingHigh = true), SESSION_ID))
        assertFalse(helper.showCombinedAlert(combinedResult(CombinedThreatLevel.DANGER), SESSION_ID))
        assertFalse(helper.showCombinedAlert(combinedResult(CombinedThreatLevel.WARNING, phishingHigh = true), SESSION_ID))
    }

    @Test
    fun `H106 expired incident resets rank and allows lower severity to post again`() {
        stubPostNotification(result = true)

        fieldMap<String, Int>("lastIncidentRank")[sessionKey()] = 3
        fieldMap<String, Long>("lastIncidentActivityTime")[sessionKey()] = System.currentTimeMillis() - 120_000L
        fieldMap<String, Long>("lastAlertTime")["incident|$SESSION_ID|WARNING|${NotificationHelper.ALERT_CHANNEL_ID}"] =
            System.currentTimeMillis() - 120_000L

        assertTrue(helper.showDetectionAlert(detectionResult(ThreatLevel.WARNING), SESSION_ID))
        assertEquals(2, fieldMap<String, Int>("lastIncidentRank")[sessionKey()])
    }

    @Test
    fun `H108 continuous observations keep incident alive and prevent downgrade repost`() {
        stubPostNotification(result = true)

        fieldMap<String, Int>("lastIncidentRank")[sessionKey()] = 3
        fieldMap<String, Long>("lastIncidentActivityTime")[sessionKey()] = System.currentTimeMillis() - 5_000L
        fieldMap<String, Long>("lastAlertTime")["incident|$SESSION_ID|WARNING|${NotificationHelper.ALERT_CHANNEL_ID}"] =
            System.currentTimeMillis() - 31_000L

        assertFalse(helper.showDetectionAlert(detectionResult(ThreatLevel.WARNING), SESSION_ID))
        assertEquals(3, fieldMap<String, Int>("lastIncidentRank")[sessionKey()])
    }

    @Test
    fun `H114 disabled channel makes postNotification fail without notify`() {
        val disabledChannel = mockk<NotificationChannel>()
        every { disabledChannel.importance } returns NotificationManager.IMPORTANCE_NONE
        every { notificationManager.getNotificationChannel(NotificationHelper.ALERT_CHANNEL_ID) } returns disabledChannel

        val posted = invokePrivatePostNotification(NotificationHelper.ALERT_CHANNEL_ID)

        assertFalse(posted)
        verify(exactly = 0) { notificationManagerCompat.notify(any(), any()) }
    }

    @Test
    fun `H116 warning cooldown is isolated by channel between deepfake and phishing alerts`() {
        stubPostNotification(result = true)

        assertTrue(helper.showDetectionAlert(detectionResult(ThreatLevel.WARNING), SESSION_ID))
        assertTrue(helper.showCombinedAlert(combinedResult(CombinedThreatLevel.WARNING, phishingHigh = true), SESSION_ID))
    }

    @Test
    fun `H124 and H159 onIncidentSafe clears incident state and cancels posted notifications`() {
        fieldMap<String, Int>("lastIncidentRank")[sessionKey()] = 4
        fieldMap<String, Long>("lastIncidentActivityTime")[sessionKey()] = System.currentTimeMillis()
        fieldMap<String, Long>("lastAlertTime")["incident|$SESSION_ID|WARNING|${NotificationHelper.ALERT_CHANNEL_ID}"] = 123L
        fieldSet<String>("deliveredSlots").add("incident|$SESSION_ID|WARNING|${NotificationHelper.ALERT_CHANNEL_ID}")
        fieldMap<String, MutableList<Int>>("postedNotificationIds")[sessionKey()] = mutableListOf(11, 12)

        helper.onIncidentSafe(SESSION_ID)

        assertFalse(fieldMap<String, Int>("lastIncidentRank").containsKey(sessionKey()))
        assertFalse(fieldMap<String, Long>("lastIncidentActivityTime").containsKey(sessionKey()))
        assertFalse(fieldMap<String, Long>("lastAlertTime").keys.any { it.startsWith("incident|$SESSION_ID|") })
        assertFalse(fieldSet<String>("deliveredSlots").any { it.startsWith("incident|$SESSION_ID|") })
        assertFalse(fieldMap<String, MutableList<Int>>("postedNotificationIds").containsKey(sessionKey()))
        verify { notificationManagerCompat.cancel(11) }
        verify { notificationManagerCompat.cancel(12) }
    }

    @Test
    fun `H163 downgrade success posts replacement and cancels stale notification ids`() {
        stubPostNotification(result = true)
        fieldMap<String, Int>("lastIncidentRank")[sessionKey()] = 4
        fieldMap<String, Long>("lastIncidentActivityTime")[sessionKey()] = System.currentTimeMillis()
        fieldMap<String, MutableList<Int>>("postedNotificationIds")[sessionKey()] = mutableListOf(21, 22)
        fieldMap<String, Long>("lastAlertTime")["incident|$SESSION_ID|CRITICAL|${NotificationHelper.COMBINED_CHANNEL_ID}"] = 500L
        fieldSet<String>("deliveredSlots").add("incident|$SESSION_ID|CRITICAL|${NotificationHelper.COMBINED_CHANNEL_ID}")

        helper.onIncidentDowngrade(SESSION_ID, combinedResult(CombinedThreatLevel.DANGER))

        assertEquals(3, fieldMap<String, Int>("lastIncidentRank")[sessionKey()])
        verify { notificationManagerCompat.cancel(21) }
        verify { notificationManagerCompat.cancel(22) }
    }

    @Test
    fun `H165 downgrade failure rolls back snapshot and keeps old notification ids`() {
        stubPostNotification(result = false)
        fieldMap<String, Int>("lastIncidentRank")[sessionKey()] = 4
        fieldMap<String, Long>("lastIncidentActivityTime")[sessionKey()] = 4_000L
        fieldMap<String, Long>("lastAlertTime")["incident|$SESSION_ID|CRITICAL|${NotificationHelper.COMBINED_CHANNEL_ID}"] = 9_999L
        fieldSet<String>("deliveredSlots").add("incident|$SESSION_ID|CRITICAL|${NotificationHelper.COMBINED_CHANNEL_ID}")
        fieldMap<String, MutableList<Int>>("postedNotificationIds")[sessionKey()] = mutableListOf(31, 32)

        helper.onIncidentDowngrade(SESSION_ID, combinedResult(CombinedThreatLevel.DANGER))

        assertEquals(4, fieldMap<String, Int>("lastIncidentRank")[sessionKey()])
        assertEquals(
            9_999L,
            fieldMap<String, Long>("lastAlertTime")["incident|$SESSION_ID|CRITICAL|${NotificationHelper.COMBINED_CHANNEL_ID}"],
        )
        assertTrue(fieldSet<String>("deliveredSlots").contains("incident|$SESSION_ID|CRITICAL|${NotificationHelper.COMBINED_CHANNEL_ID}"))
        assertEquals(listOf(31, 32), fieldMap<String, MutableList<Int>>("postedNotificationIds")[sessionKey()])
        verify(exactly = 0) { notificationManagerCompat.cancel(31) }
        verify(exactly = 0) { notificationManagerCompat.cancel(32) }
    }

    @Test
    fun `H180 ownership re-check skips stale post after another attempt takes the slot`() {
        stubPostNotification(result = true)
        installOwnershipRaceHook {
            val owners = fieldMap<String, Long>("cooldownAttemptOwner")
            owners.keys.toList().forEach { owners[it] = Long.MIN_VALUE }
        }

        val posted = helper.showCombinedAlert(combinedResult(CombinedThreatLevel.WARNING, phishingHigh = true), SESSION_ID)

        assertFalse(posted)
        assertTrue(fieldSet<String>("deliveredSlots").isEmpty())
    }

    // showServiceError는 NotificationCompat.Builder 실제 구현 의존 — unit test stub에서는
    // 검증 불가. integration test로 따로 확인 필요.

    private fun stubPostNotification(result: Boolean) {
        every {
            helper["postNotification"](
                any<String>(),
                any<String>(),
                any<String>(),
                any<Int>(),
                any<String>(),
            )
        } returns result
    }

    private fun installOwnershipRaceHook(hook: () -> Unit) {
        val field = NotificationHelper::class.java.getDeclaredField("beforeOwnershipRecheck")
        field.isAccessible = true
        field.set(helper, hook)
    }

    private fun invokePrivatePostNotification(channelId: String): Boolean {
        val method = NotificationHelper::class.java.getDeclaredMethod(
            "postNotification",
            String::class.java,
            String::class.java,
            String::class.java,
            Int::class.javaPrimitiveType,
            String::class.java,
        )
        method.isAccessible = true
        return method.invoke(
            helper,
            channelId,
            "title",
            "text",
            0,
            sessionKey(),
        ) as Boolean
    }

    @Suppress("UNCHECKED_CAST")
    private fun <K, V> fieldMap(name: String): MutableMap<K, V> =
        declaredField(name).get(helper) as MutableMap<K, V>

    @Suppress("UNCHECKED_CAST")
    private fun <T> fieldSet(name: String): MutableSet<T> =
        declaredField(name).get(helper) as MutableSet<T>

    private fun declaredField(name: String): Field =
        NotificationHelper::class.java.getDeclaredField(name).apply { isAccessible = true }

    private fun sessionKey(): String = "session|$SESSION_ID"

    private fun detectionResult(level: ThreatLevel): AggregatedResult = AggregatedResult(
        threatLevel = level,
        averageFakeScore = when (level) {
            ThreatLevel.SAFE -> 0.1f
            ThreatLevel.CAUTION -> 0.61f
            ThreatLevel.WARNING -> 0.75f
            ThreatLevel.DANGER -> 0.95f
        },
        latestResult = DetectionResult(
            fakeScore = 0.9f,
            realScore = 0.1f,
            confidence = 0.9f,
            latencyMs = 42L,
        ),
        consecutiveHighCount = if (level.ordinal >= ThreatLevel.WARNING.ordinal) 3 else 0,
    )

    private fun combinedResult(
        level: CombinedThreatLevel,
        phishingHigh: Boolean = false,
    ): CombinedThreatResult = CombinedThreatResult(
        combinedThreatLevel = level,
        deepfakeResult = detectionResult(
            when (level) {
                CombinedThreatLevel.SAFE -> ThreatLevel.SAFE
                CombinedThreatLevel.CAUTION -> ThreatLevel.CAUTION
                CombinedThreatLevel.WARNING -> ThreatLevel.WARNING
                CombinedThreatLevel.DANGER -> ThreatLevel.DANGER
                CombinedThreatLevel.CRITICAL -> ThreatLevel.DANGER
            }
        ),
        deepfakeThreatLevel = when (level) {
            CombinedThreatLevel.SAFE -> ThreatLevel.SAFE
            CombinedThreatLevel.CAUTION -> ThreatLevel.CAUTION
            CombinedThreatLevel.WARNING -> ThreatLevel.WARNING
            CombinedThreatLevel.DANGER -> ThreatLevel.DANGER
            CombinedThreatLevel.CRITICAL -> ThreatLevel.DANGER
        },
        phishingResult = if (phishingHigh) {
            PhishingResult(
                score = 0.8f,
                matchedKeywords = emptyList(),
                matchedPhrases = listOf("안전한 계좌로 이체"),
                threatLevel = PhishingThreatLevel.HIGH,
                transcription = "fixture",
            )
        } else {
            null
        },
        phishingScore = if (phishingHigh) 0.8f else 0f,
        matchedKeywords = emptyList(),
        matchedPhrases = if (phishingHigh) listOf("안전한 계좌로 이체") else emptyList(),
        sttStatus = SttStatus.LISTENING,
        sttAvailable = true,
        transcription = "fixture",
    )
    companion object {
        private const val SESSION_ID = "session-under-test"
    }
}
