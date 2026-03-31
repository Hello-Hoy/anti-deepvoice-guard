package com.deepvoiceguard.app

import android.app.Application
import com.deepvoiceguard.app.storage.CleanupWorker
import dagger.hilt.android.HiltAndroidApp

@HiltAndroidApp
class DeepVoiceGuardApp : Application() {
    override fun onCreate() {
        super.onCreate()
        CleanupWorker.schedule(this)
    }
}
