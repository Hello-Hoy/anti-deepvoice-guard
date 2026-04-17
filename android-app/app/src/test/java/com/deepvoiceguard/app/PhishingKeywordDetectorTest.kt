package com.deepvoiceguard.app

import android.content.Context
import android.content.res.AssetManager
import com.deepvoiceguard.app.phishing.PhishingKeywordDetector
import com.deepvoiceguard.app.phishing.model.PhishingThreatLevel
import io.mockk.every
import io.mockk.mockk
import java.io.ByteArrayInputStream
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class PhishingKeywordDetectorTest {

    private lateinit var detector: PhishingKeywordDetector

    @Before
    fun setUp() {
        detector = PhishingKeywordDetector(assetBackedContext())
    }

    @Test
    fun `phishing fixtures are classified as warning or higher`() {
        phishingFixtures().forEach { fixture ->
            val result = detector.analyze(fixture)
            assertTrue(
                "Expected warning-or-higher for phishing fixture: $fixture, got ${result.threatLevel} (${result.score})",
                isPositive(result.threatLevel),
            )
        }
    }

    @Test
    fun `benign fixtures must not hit HIGH`() {
        // MEDIUM은 오탐 여지가 있는 경계 — HIGH 로 오인하지만 않으면 허용.
        // 엄격한 오탐 측정은 F1 score test가 담당.
        benignFixtures().forEach { fixture ->
            val result = detector.analyze(fixture)
            assertTrue(
                "benign 이 HIGH로 오분류: $fixture (${result.score})",
                result.threatLevel != PhishingThreatLevel.HIGH,
            )
        }
    }

    @Test
    fun `fixture suite keeps f1 score above threshold`() {
        val positives = phishingFixtures().map { true to detector.analyze(it).threatLevel }
        val negatives = benignFixtures().map { false to detector.analyze(it).threatLevel }
        val labeled = positives + negatives

        val tp = labeled.count { (expected, actual) -> expected && isPositive(actual) }.toDouble()
        val fp = labeled.count { (expected, actual) -> !expected && isPositive(actual) }.toDouble()
        val fn = labeled.count { (expected, actual) -> expected && !isPositive(actual) }.toDouble()

        val precision = if (tp + fp == 0.0) 0.0 else tp / (tp + fp)
        val recall = if (tp + fn == 0.0) 0.0 else tp / (tp + fn)
        val f1 = if (precision + recall == 0.0) 0.0 else (2 * precision * recall) / (precision + recall)

        assertTrue("Expected F1 >= 0.75, got $f1 (precision=$precision, recall=$recall)", f1 >= 0.75)
    }

    private fun isPositive(level: PhishingThreatLevel): Boolean =
        level == PhishingThreatLevel.MEDIUM || level == PhishingThreatLevel.HIGH

    private fun assetBackedContext(): Context {
        val context = mockk<Context>()
        val assetManager = mockk<AssetManager>()
        every { context.assets } returns assetManager
        every { assetManager.open("phishing_keywords.json") } answers {
            ByteArrayInputStream(readAssetText("phishing_keywords.json").toByteArray(StandardCharsets.UTF_8))
        }
        every { assetManager.open("phishing_phrases.json") } answers {
            ByteArrayInputStream(readAssetText("phishing_phrases.json").toByteArray(StandardCharsets.UTF_8))
        }
        return context
    }

    private fun readAssetText(fileName: String): String {
        val candidates = listOf(
            Path.of("android-app", "app", "src", "main", "assets", fileName),
            Path.of("app", "src", "main", "assets", fileName),
            Path.of("src", "main", "assets", fileName),
        )
        val path = candidates.firstOrNull { Files.exists(it) }
            ?: error("Unable to locate asset fixture for $fileName from ${Path.of("").toAbsolutePath()}")
        return String(Files.readAllBytes(path), StandardCharsets.UTF_8)
    }

    private fun phishingFixtures(): List<String> = listOf(
        "여기는 검찰청 수사관입니다 사건 번호를 확인하시고 안전한 계좌로 이체하십시오",
        "중앙지검 수사관인데 출석 요구가 있으니 지금 바로 연락을 유지하세요",
        "경찰청 수사관입니다 명의도용 사건이라 수사에 협조하시고 오늘 안에 처리해야 합니다",
        "금융감독원 직원입니다 계좌가 범죄에 연루되어 국가 안전 계좌로 송금해야 합니다",
        "검사입니다 체포 영장이 발부되기 전에 자금을 보호하기 위해 임시 보호 계좌로 보내세요",
        "저금리로 대환해 드리니 수수료를 먼저 입금해 주셔야 합니다",
        "구속 영장이 발부될 수 있으니 오늘 안에 처리하고 현금을 인출해서 전달하세요",
        "국세청 담당자입니다 벌금 미납 건으로 지금 즉시 이체하셔야 불이익이 없습니다",
        "법원 담당자인데 개인정보가 유출되어 주민번호 뒷자리를 확인해야 합니다",
        "경찰관입니다 대포통장 사건이어서 ATM에서 현금 인출 후 보관해 두셔야 합니다",
        "금감원 직원인데 보증금을 납부해야 저금리로 대환이 가능합니다",
        "수사 담당자입니다 범죄 혐의가 있어 사건 번호를 확인하고 계좌번호를 말씀해 주세요",
        "금융위원회 직원인데 시간이 얼마 없으니 안전계좌로 빨리 송금하셔야 합니다",
        "검사입니다 본인 확인을 위해 OTP와 계좌번호를 말씀하시고 명의가 도용되었는지 확인합시다",
        "중앙지검 수사관입니다 고소 사건으로 출석 요구가 있으며 오늘 안에 처리하지 않으면 처벌됩니다",
    )

    private fun benignFixtures(): List<String> = listOf(
        "고객님 카드 배송 일정 안내드립니다 오늘 오후에 배송 예정입니다",
        "보험 상담 예약 시간을 확인하려고 연락드렸습니다",
        "은행 본인 확인을 위해 생년월일만 확인하겠습니다",
        "대출 심사 결과 안내드리며 추가 서류 제출이 필요합니다",
        "체크카드 재발급 접수가 완료되어 우편 배송 예정입니다",
        "은행 앱 점검 시간 안내로 내일 새벽 서비스가 일시 중단됩니다",
        "정기 적금 만기 후 재예치 여부를 확인 부탁드립니다",
        "고객센터 만족도 조사 참여 여부만 여쭙겠습니다",
        "보험금 청구 서류 접수 상태를 안내드립니다",
        "예금 만기 안내로 연락드렸으며 금리 변경 내용을 설명드리겠습니다",
    )
}
