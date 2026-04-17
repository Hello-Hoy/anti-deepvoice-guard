package com.deepvoiceguard.app.storage

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(entities = [DetectionEntity::class], version = 3, exportSchema = true)
abstract class DetectionDatabase : RoomDatabase() {
    abstract fun detectionDao(): DetectionDao

    companion object {
        @Volatile
        private var INSTANCE: DetectionDatabase? = null

        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE detections ADD COLUMN callSessionId TEXT")
            }
        }

        val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE detections ADD COLUMN phishingScore REAL NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE detections ADD COLUMN phishingKeywords TEXT")
                db.execSQL("ALTER TABLE detections ADD COLUMN transcription TEXT")
                db.execSQL("ALTER TABLE detections ADD COLUMN combinedThreatLevel TEXT NOT NULL DEFAULT 'SAFE'")
                db.execSQL("ALTER TABLE detections ADD COLUMN sttAvailable INTEGER NOT NULL DEFAULT 1")
            }
        }

        fun getInstance(context: Context): DetectionDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    DetectionDatabase::class.java,
                    "deepvoice_guard.db",
                )
                    .addMigrations(MIGRATION_1_2, MIGRATION_2_3)
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}
