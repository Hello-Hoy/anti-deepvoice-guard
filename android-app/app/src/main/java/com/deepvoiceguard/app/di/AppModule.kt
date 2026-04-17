package com.deepvoiceguard.app.di

import android.content.Context
import com.deepvoiceguard.app.inference.CombinedThreatAggregator
import com.deepvoiceguard.app.phishing.PhishingKeywordDetector
import com.deepvoiceguard.app.stt.SttCapabilityChecker
import com.deepvoiceguard.app.storage.DetectionDao
import com.deepvoiceguard.app.storage.DetectionDatabase
import com.deepvoiceguard.app.storage.EncryptedStorage
import com.deepvoiceguard.app.storage.SettingsRepository
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): DetectionDatabase =
        DetectionDatabase.getInstance(context)

    @Provides
    fun provideDetectionDao(db: DetectionDatabase): DetectionDao =
        db.detectionDao()

    @Provides
    @Singleton
    fun provideEncryptedStorage(@ApplicationContext context: Context): EncryptedStorage =
        EncryptedStorage(context)

    @Provides
    @Singleton
    fun provideSettingsRepository(@ApplicationContext context: Context): SettingsRepository =
        SettingsRepository(context)

    @Provides
    @Singleton
    fun provideSttCapabilityChecker(@ApplicationContext context: Context): SttCapabilityChecker =
        SttCapabilityChecker(context)

    @Provides
    @Singleton
    fun providePhishingKeywordDetector(@ApplicationContext context: Context): PhishingKeywordDetector =
        PhishingKeywordDetector(context)

    @Provides
    @Singleton
    fun provideCombinedThreatAggregator(): CombinedThreatAggregator =
        CombinedThreatAggregator()
}
