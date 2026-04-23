package com.deepvoiceguard.app.storage

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

data class AppSettings(
    val useOnDevice: Boolean = true,
    val serverUrl: String = "http://192.168.0.100:8000",
    val threshold: Float = 0.9f,
    val autoStartOnCall: Boolean = true,
    val autoStartOnBoot: Boolean = false,
    // --- v3 STT/피싱 설정 ---
    val sttEnabled: Boolean = false,                // STT 기능 (기본 OFF — 동의 후 활성화)
    val sttConsentGiven: Boolean = false,            // STT 음성 전송 동의 (별도)
    val phishingDetectionEnabled: Boolean = true,    // 피싱 키워드 탐지
    val transcriptStorageEnabled: Boolean = false,   // 전사 텍스트 로컬 저장 동의
    val hybridModeEnabled: Boolean = false,          // 하이브리드 서버 모드
    val dataConsentLevel: String = "none",           // "none", "metadata", "full"
    val demoMode: Boolean = false,                   // 데모 모드
    val liveNarrowbandEnabled: Boolean = false,      // v4+ 모델은 raw PCM 입력 기준 학습됐으므로 기본 비활성화. true로 켜면 narrowband 적용(레거시 호환).
)

class SettingsRepository(private val context: Context) {

    private object Keys {
        val USE_ON_DEVICE = booleanPreferencesKey("use_on_device")
        val SERVER_URL = stringPreferencesKey("server_url")
        val THRESHOLD = floatPreferencesKey("threshold")
        val AUTO_START_ON_CALL = booleanPreferencesKey("auto_start_on_call")
        val AUTO_START_ON_BOOT = booleanPreferencesKey("auto_start_on_boot")
        val STT_ENABLED = booleanPreferencesKey("stt_enabled")
        val STT_CONSENT_GIVEN = booleanPreferencesKey("stt_consent_given")
        val PHISHING_DETECTION_ENABLED = booleanPreferencesKey("phishing_detection_enabled")
        val TRANSCRIPT_STORAGE_ENABLED = booleanPreferencesKey("transcript_storage_enabled")
        val HYBRID_MODE_ENABLED = booleanPreferencesKey("hybrid_mode_enabled")
        val DATA_CONSENT_LEVEL = stringPreferencesKey("data_consent_level")
        val DEMO_MODE = booleanPreferencesKey("demo_mode")
        val LIVE_NARROWBAND_ENABLED = booleanPreferencesKey("live_narrowband_enabled")
    }

    val settings: Flow<AppSettings> = context.dataStore.data.map { prefs ->
        AppSettings(
            useOnDevice = prefs[Keys.USE_ON_DEVICE] ?: true,
            serverUrl = prefs[Keys.SERVER_URL] ?: "http://192.168.0.100:8000",
            threshold = prefs[Keys.THRESHOLD] ?: 0.9f,
            autoStartOnCall = prefs[Keys.AUTO_START_ON_CALL] ?: true,
            autoStartOnBoot = prefs[Keys.AUTO_START_ON_BOOT] ?: false,
            sttEnabled = prefs[Keys.STT_ENABLED] ?: false,
            sttConsentGiven = prefs[Keys.STT_CONSENT_GIVEN] ?: false,
            phishingDetectionEnabled = prefs[Keys.PHISHING_DETECTION_ENABLED] ?: true,
            transcriptStorageEnabled = prefs[Keys.TRANSCRIPT_STORAGE_ENABLED] ?: false,
            hybridModeEnabled = prefs[Keys.HYBRID_MODE_ENABLED] ?: false,
            dataConsentLevel = prefs[Keys.DATA_CONSENT_LEVEL] ?: "none",
            demoMode = prefs[Keys.DEMO_MODE] ?: false,
            liveNarrowbandEnabled = prefs[Keys.LIVE_NARROWBAND_ENABLED] ?: false,
        )
    }

    suspend fun setLiveNarrowbandEnabled(value: Boolean) {
        context.dataStore.edit { it[Keys.LIVE_NARROWBAND_ENABLED] = value }
    }

    suspend fun setUseOnDevice(value: Boolean) {
        context.dataStore.edit { it[Keys.USE_ON_DEVICE] = value }
    }

    suspend fun setServerUrl(value: String) {
        context.dataStore.edit { it[Keys.SERVER_URL] = value }
    }

    suspend fun setThreshold(value: Float) {
        context.dataStore.edit { it[Keys.THRESHOLD] = value }
    }

    suspend fun setAutoStartOnCall(value: Boolean) {
        context.dataStore.edit { it[Keys.AUTO_START_ON_CALL] = value }
        // PhoneStateReceiver와 BootReceiver가 사용하는 SharedPreferences에도 동기화
        context.getSharedPreferences("phone_state_prefs", Context.MODE_PRIVATE)
            .edit().putBoolean("auto_start_on_call", value).apply()
    }

    suspend fun setAutoStartOnBoot(value: Boolean) {
        context.dataStore.edit { it[Keys.AUTO_START_ON_BOOT] = value }
        syncToSharedPrefs("auto_start_on_boot", value)
    }

    suspend fun setSttEnabled(value: Boolean) {
        context.dataStore.edit { it[Keys.STT_ENABLED] = value }
        syncToSharedPrefs("stt_enabled", value)
    }

    suspend fun setSttConsentGiven(value: Boolean) {
        context.dataStore.edit { it[Keys.STT_CONSENT_GIVEN] = value }
        syncToSharedPrefs("stt_consent_given", value)
    }

    suspend fun setPhishingDetectionEnabled(value: Boolean) {
        context.dataStore.edit { it[Keys.PHISHING_DETECTION_ENABLED] = value }
    }

    suspend fun setTranscriptStorageEnabled(value: Boolean) {
        context.dataStore.edit { it[Keys.TRANSCRIPT_STORAGE_ENABLED] = value }
    }

    suspend fun setHybridModeEnabled(value: Boolean) {
        context.dataStore.edit { it[Keys.HYBRID_MODE_ENABLED] = value }
    }

    suspend fun setDataConsentLevel(value: String) {
        context.dataStore.edit { it[Keys.DATA_CONSENT_LEVEL] = value }
    }

    suspend fun setDemoMode(value: Boolean) {
        context.dataStore.edit { it[Keys.DEMO_MODE] = value }
    }

    private fun syncToSharedPrefs(key: String, value: Boolean) {
        context.getSharedPreferences("phone_state_prefs", Context.MODE_PRIVATE)
            .edit().putBoolean(key, value).apply()
    }
}
