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
    val threshold: Float = 0.7f,
    val autoStartOnCall: Boolean = true,
    val autoStartOnBoot: Boolean = false,
)

class SettingsRepository(private val context: Context) {

    private object Keys {
        val USE_ON_DEVICE = booleanPreferencesKey("use_on_device")
        val SERVER_URL = stringPreferencesKey("server_url")
        val THRESHOLD = floatPreferencesKey("threshold")
        val AUTO_START_ON_CALL = booleanPreferencesKey("auto_start_on_call")
        val AUTO_START_ON_BOOT = booleanPreferencesKey("auto_start_on_boot")
    }

    val settings: Flow<AppSettings> = context.dataStore.data.map { prefs ->
        AppSettings(
            useOnDevice = prefs[Keys.USE_ON_DEVICE] ?: true,
            serverUrl = prefs[Keys.SERVER_URL] ?: "http://192.168.0.100:8000",
            threshold = prefs[Keys.THRESHOLD] ?: 0.7f,
            autoStartOnCall = prefs[Keys.AUTO_START_ON_CALL] ?: true,
            autoStartOnBoot = prefs[Keys.AUTO_START_ON_BOOT] ?: false,
        )
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
        context.getSharedPreferences("phone_state_prefs", Context.MODE_PRIVATE)
            .edit().putBoolean("auto_start_on_boot", value).apply()
    }
}
