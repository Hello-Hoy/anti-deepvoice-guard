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
)

class SettingsRepository(private val context: Context) {

    private object Keys {
        val USE_ON_DEVICE = booleanPreferencesKey("use_on_device")
        val SERVER_URL = stringPreferencesKey("server_url")
        val THRESHOLD = floatPreferencesKey("threshold")
    }

    val settings: Flow<AppSettings> = context.dataStore.data.map { prefs ->
        AppSettings(
            useOnDevice = prefs[Keys.USE_ON_DEVICE] ?: true,
            serverUrl = prefs[Keys.SERVER_URL] ?: "http://192.168.0.100:8000",
            threshold = prefs[Keys.THRESHOLD] ?: 0.7f,
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
}
