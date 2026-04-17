package com.deepvoiceguard.app.ui

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.PlayCircle
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.deepvoiceguard.app.ui.screens.AboutScreen
import com.deepvoiceguard.app.ui.screens.DemoScreen
import com.deepvoiceguard.app.ui.screens.HistoryScreen
import com.deepvoiceguard.app.ui.screens.HomeScreen
import com.deepvoiceguard.app.ui.screens.SettingsScreen

enum class Screen(val route: String, val label: String, val icon: ImageVector) {
    Home("home", "Home", Icons.Default.Home),
    Demo("demo", "Demo", Icons.Default.PlayCircle),
    History("history", "History", Icons.Default.History),
    Settings("settings", "Settings", Icons.Default.Settings),
    About("about", "About", Icons.Default.Info),
}

/** Demo asset 존재 여부 확인 — 적어도 하나의 시나리오에 WAV가 있어야 Demo 탭 노출. */
private fun hasAnyDemoAudio(context: android.content.Context): Boolean {
    return try {
        val files = context.assets.list("demo") ?: return false
        files.any { it.endsWith(".wav") }
    } catch (_: Exception) {
        false
    }
}

@Composable
fun DeepVoiceGuardApp() {
    val context = LocalContext.current
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = navBackStackEntry?.destination

    // Demo 탭은 WAV asset이 패키징된 빌드에서만 노출.
    val showDemoTab = remember(context) { hasAnyDemoAudio(context) }
    val visibleScreens = remember(showDemoTab) {
        Screen.entries.filter { it != Screen.Demo || showDemoTab }
    }

    Scaffold(
        bottomBar = {
            NavigationBar {
                visibleScreens.forEach { screen ->
                    NavigationBarItem(
                        icon = { Icon(screen.icon, contentDescription = screen.label) },
                        label = { Text(screen.label) },
                        selected = currentDestination?.hierarchy?.any { it.route == screen.route } == true,
                        onClick = {
                            navController.navigate(screen.route) {
                                popUpTo(navController.graph.findStartDestination().id) {
                                    saveState = true
                                }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                    )
                }
            }
        },
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Home.route,
            modifier = Modifier.padding(innerPadding),
        ) {
            composable(Screen.Home.route) { HomeScreen() }
            // Demo는 WAV가 있을 때만 navigation 그래프에 등록.
            if (showDemoTab) {
                composable(Screen.Demo.route) { DemoScreen() }
            }
            composable(Screen.History.route) { HistoryScreen() }
            composable(Screen.Settings.route) { SettingsScreen() }
            composable(Screen.About.route) { AboutScreen() }
        }
    }
}
