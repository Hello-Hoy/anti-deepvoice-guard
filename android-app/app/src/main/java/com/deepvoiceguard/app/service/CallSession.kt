package com.deepvoiceguard.app.service

import java.util.UUID

enum class CallDirection { INCOMING, OUTGOING, UNKNOWN }

data class CallSession(
    val sessionId: String = UUID.randomUUID().toString(),
    val startTimeMs: Long = System.currentTimeMillis(),
    val phoneNumber: String? = null,
    val direction: CallDirection = CallDirection.UNKNOWN,
    var endTimeMs: Long? = null,
) {
    val isActive: Boolean get() = endTimeMs == null

    fun end(): CallSession = copy(endTimeMs = System.currentTimeMillis())
}
