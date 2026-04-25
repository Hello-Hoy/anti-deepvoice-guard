# Feature 3 — 알뜰폰 사업자 브랜드 테마 (2가지 가상 브랜드)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 사용자가 Settings 에서 3가지 브랜드(기본 + 2개 가상 알뜰폰) 중 하나를 선택하면 앱 색상 팔레트, 상단 브랜드 헤더(로고+이름)가 즉시 바뀐다. 데모 시 "이 앱이 다양한 알뜰폰 사업자 브랜드로 OEM 가능함"을 시연.

**Architecture:**
- `enum CarrierBrand` 가 모든 브랜드 메타데이터를 가짐 (key, displayName, logo res, primary/secondary color). 별도 BrandSpec 클래스 없음 — enum 자체로 충분.
- `Theme.kt` 의 `DeepVoiceGuardTheme` 가 `CarrierBrand` 인자를 받음. `CompositionLocalProvider(LocalCarrierBrand provides brand)` 로 자식 컴포넌트가 brand 메타데이터 접근 가능.
- `MainActivity` 가 `SettingsRepository` 를 통해 `settings.carrierBrand` 를 구독 → Theme 에 전달. **DataStore 인스턴스화 정책**: top-level `preferencesDataStore` delegate (`SettingsRepository.kt:14`) 가 process-singleton 을 보장하므로 `SettingsRepository(context)` 를 직접 생성해도 안전. 본 plan 은 기존 `SettingsScreen.kt` 와 동일한 `remember { SettingsRepository(context) }` 패턴 사용 (이미 검증된 fallback). MainActivity 를 `@AndroidEntryPoint` 로 변환하는 작업은 후속 cleanup task — 본 plan 범위 밖.
- `BrandHeader` Composable 이 `LocalCarrierBrand.current` 를 읽어 로고+이름 표시 — HomeScreen 상단에 삽입.
- 브랜드명은 **완전 가상**: `Default` / `Nova 모바일` / `Lumi 모바일` — 실존 통신사 (Mint/Aqua 등) 와 명백히 구별되도록 영어+한글 조합 + 색상도 차별화.
- Settings UI 는 **단일 선택 RadioButton** 그룹 (Switch 3개는 상호배타 의미 약함).

**Tech Stack:** Kotlin, Jetpack Compose Material 3 ColorScheme, Vector drawable, Hilt @Inject SettingsRepository.

**Base branch:** `android/real-device-verify`. Plan 1, 2 와 병렬 작업 가능 — `HomeScreen.kt` (Plan 1 도 수정), `SettingsRepository.kt` / `SettingsScreen.kt` (Plan 2 도 수정) 충돌 가능. **머지 시 마지막에 적용 권장**.

---

## File Structure

| File | 책임 |
|---|---|
| `android-app/app/src/main/java/com/deepvoiceguard/app/ui/theme/CarrierBrand.kt` (신규) | enum + 메타데이터 + fromKey + lightScheme/darkScheme |
| `android-app/app/src/main/java/com/deepvoiceguard/app/ui/theme/Theme.kt` | DeepVoiceGuardTheme 가 brand 인자 받게 변경 + LocalCarrierBrand provider |
| `android-app/app/src/main/res/drawable/logo_default.xml` (신규) | 기본 placeholder |
| `android-app/app/src/main/res/drawable/logo_nova.xml` (신규) | Nova 모바일 가상 로고 |
| `android-app/app/src/main/res/drawable/logo_lumi.xml` (신규) | Lumi 모바일 가상 로고 |
| `android-app/app/src/main/java/com/deepvoiceguard/app/storage/SettingsRepository.kt` | `carrierBrand: String` 필드 + Keys + setter |
| `android-app/app/src/main/java/com/deepvoiceguard/app/ui/MainActivity.kt` | Hilt @Inject SettingsRepository → settings.carrierBrand → Theme |
| `android-app/app/src/main/java/com/deepvoiceguard/app/ui/components/BrandHeader.kt` (신규) | LocalCarrierBrand 구독해 로고+이름 표시 |
| `android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/HomeScreen.kt` | BrandHeader 추가 (HomeScreen 상단만) |
| `android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/SettingsScreen.kt` | "브랜드 (데모용)" 카드 + RadioButton 그룹 |
| `android-app/app/src/test/java/com/deepvoiceguard/app/ui/CarrierBrandTest.kt` (신규) | enum/string 매핑 단위 테스트 |

---

## Phase 0 — Branch & 백업

- [ ] **Step 0.1: 새 feature branch**

```bash
git checkout android/real-device-verify
git checkout -b feature/brand-themes
```

- [ ] **Step 0.2: 원격 push**

```bash
git push -u origin feature/brand-themes
```

---

## Phase 1 — CarrierBrand enum + 테스트

### Task 1: 데이터 모델

**Files:**
- Create: `android-app/app/src/main/java/com/deepvoiceguard/app/ui/theme/CarrierBrand.kt`
- Test: `android-app/app/src/test/java/com/deepvoiceguard/app/ui/CarrierBrandTest.kt`

- [ ] **Step 1.1: 테스트 작성**

```kotlin
package com.deepvoiceguard.app.ui

import com.deepvoiceguard.app.ui.theme.CarrierBrand
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CarrierBrandTest {
    @Test
    fun `default brand round-trip via key`() {
        assertEquals(CarrierBrand.DEFAULT, CarrierBrand.fromKey("default"))
    }

    @Test
    fun `nova brand round-trip`() {
        assertEquals(CarrierBrand.NOVA, CarrierBrand.fromKey("nova"))
    }

    @Test
    fun `lumi brand round-trip`() {
        assertEquals(CarrierBrand.LUMI, CarrierBrand.fromKey("lumi"))
    }

    @Test
    fun `unknown key falls back to default`() {
        assertEquals(CarrierBrand.DEFAULT, CarrierBrand.fromKey("unknown"))
        assertEquals(CarrierBrand.DEFAULT, CarrierBrand.fromKey(""))
        assertEquals(CarrierBrand.DEFAULT, CarrierBrand.fromKey(null))
    }

    @Test
    fun `every brand has display name and unique key`() {
        val keys = CarrierBrand.values().map { it.key }
        assertEquals("brand keys must be unique", keys.distinct().size, keys.size)
        for (b in CarrierBrand.values()) {
            assertTrue("brand displayName must not be blank", b.displayName.isNotBlank())
            assertTrue("brand key must not be blank", b.key.isNotBlank())
        }
    }
}
```

- [ ] **Step 1.2: 테스트 실패 확인**

```bash
cd android-app
./gradlew :app:testDebugUnitTest --tests "com.deepvoiceguard.app.ui.CarrierBrandTest" 2>&1 | tail -10
```

기대: unresolved reference `CarrierBrand`.

- [ ] **Step 1.3: enum 작성**

```kotlin
package com.deepvoiceguard.app.ui.theme

import androidx.annotation.DrawableRes
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.ui.graphics.Color
import com.deepvoiceguard.app.R

/**
 * 데모 시연용 가상 알뜰폰 사업자 브랜드.
 *
 * **법적 안전 원칙**: 실존 통신사/MVNO 명칭(Mint, Aqua, KT, LG U+ 등)을 직접 쓰지 않는다.
 * `Nova 모바일`, `Lumi 모바일` 은 완전 가상 — 동일 명칭의 실 사업자 검색 결과 없음 (작성 시점).
 * 데모 발표 시 "예시 OEM 변형" 으로 명시.
 */
enum class CarrierBrand(
    val key: String,
    val displayName: String,
    @DrawableRes val logoRes: Int,
    val primary: Color,
    val secondary: Color,
) {
    DEFAULT(
        key = "default",
        displayName = "Anti-DeepVoice Guard",
        logoRes = R.drawable.logo_default,
        primary = Color(0xFF1E88E5),    // 파랑 — 기본 브랜드
        secondary = Color(0xFF42A5F5),
    ),
    NOVA(
        key = "nova",
        displayName = "Nova 모바일",
        logoRes = R.drawable.logo_nova,
        primary = Color(0xFF6A1B9A),    // 보라 — 차별화
        secondary = Color(0xFFAB47BC),
    ),
    LUMI(
        key = "lumi",
        displayName = "Lumi 모바일",
        logoRes = R.drawable.logo_lumi,
        primary = Color(0xFFEF6C00),    // 주황 — 차별화
        secondary = Color(0xFFFFB74D),
    );

    fun lightScheme(): ColorScheme = lightColorScheme(
        primary = primary,
        secondary = secondary,
    )

    fun darkScheme(): ColorScheme = darkColorScheme(
        primary = primary,
        secondary = secondary,
    )

    companion object {
        fun fromKey(key: String?): CarrierBrand =
            values().firstOrNull { it.key == key } ?: DEFAULT
    }
}
```

- [ ] **Step 1.4: 로고 vector drawable 생성**

`android-app/app/src/main/res/drawable/logo_default.xml`:

```xml
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="48dp" android:height="48dp"
    android:viewportWidth="48" android:viewportHeight="48">
    <path android:fillColor="#1E88E5"
        android:pathData="M24,4 L42,14 V26 C42,36 34,42 24,44 C14,42 6,36 6,26 V14 Z" />
    <path android:fillColor="#FFFFFF" android:strokeColor="#FFFFFF"
        android:strokeWidth="3" android:strokeLineCap="round" android:strokeLineJoin="round"
        android:pathData="M18,22 L22,28 L30,18" />
</vector>
```

`android-app/app/src/main/res/drawable/logo_nova.xml` (보라 별 모티브):

```xml
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="48dp" android:height="48dp"
    android:viewportWidth="48" android:viewportHeight="48">
    <path android:fillColor="#6A1B9A"
        android:pathData="M24,4 L29,18 L44,18 L32,28 L37,42 L24,33 L11,42 L16,28 L4,18 L19,18 Z" />
</vector>
```

`android-app/app/src/main/res/drawable/logo_lumi.xml` (주황 원 모티브):

```xml
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="48dp" android:height="48dp"
    android:viewportWidth="48" android:viewportHeight="48">
    <path android:fillColor="#EF6C00"
        android:pathData="M24,8 A16,16 0 1,0 24,40 A16,16 0 1,0 24,8 Z" />
    <path android:fillColor="#FFFFFF"
        android:pathData="M24,16 A8,8 0 1,0 24,32 A8,8 0 1,0 24,16 Z" />
</vector>
```

- [ ] **Step 1.5: 테스트 통과 확인**

```bash
./gradlew :app:testDebugUnitTest --tests "com.deepvoiceguard.app.ui.CarrierBrandTest" 2>&1 | tail -10
```

기대: BUILD SUCCESSFUL, 5 passed.

- [ ] **Step 1.6: 커밋**

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/ui/theme/CarrierBrand.kt \
         android-app/app/src/main/res/drawable/logo_default.xml \
         android-app/app/src/main/res/drawable/logo_nova.xml \
         android-app/app/src/main/res/drawable/logo_lumi.xml \
         android-app/app/src/test/java/com/deepvoiceguard/app/ui/CarrierBrandTest.kt
git commit -m "feat(theme): CarrierBrand enum (Default/Nova/Lumi) + 3 vector logos

가상 알뜰폰 브랜드 — 법적 안전을 위해 실존 통신사 미사용.
- Nova 모바일 (보라/별) / Lumi 모바일 (주황/원) / Anti-DeepVoice Guard (기본 파랑/방패)
- fromKey(nullable) with default fallback
- 5 unit tests"
```

---

## Phase 2 — SettingsRepository 필드

### Task 2: brand 필드 추가

**Files:**
- Modify: `android-app/app/src/main/java/com/deepvoiceguard/app/storage/SettingsRepository.kt`

- [ ] **Step 2.1: AppSettings 에 carrierBrand: String 추가**

기존 data class 끝에 추가:

```kotlin
val carrierBrand: String = "default",  // CarrierBrand.key — UI 테마 분기
```

- [ ] **Step 2.2: Keys + reader + setter 추가**

```kotlin
// Keys object
val CARRIER_BRAND = stringPreferencesKey("carrier_brand")

// settings flow
carrierBrand = prefs[Keys.CARRIER_BRAND] ?: "default",

// setter (기존 setter 패턴 동일)
suspend fun setCarrierBrand(value: String) {
    context.dataStore.edit { it[Keys.CARRIER_BRAND] = value }
}
```

- [ ] **Step 2.3: 빌드 + 커밋**

```bash
./gradlew :app:compileDebugKotlin 2>&1 | tail -3
```

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/storage/SettingsRepository.kt
git commit -m "feat(settings): add carrierBrand field for theme selection"
```

---

## Phase 3 — Theme brand-aware

### Task 3: DeepVoiceGuardTheme 변경

**Files:**
- Modify: `android-app/app/src/main/java/com/deepvoiceguard/app/ui/theme/Theme.kt`

- [ ] **Step 3.1: 함수 시그니처 + LocalCarrierBrand 추가**

기존 `Theme.kt` 전체를 다음으로 교체:

```kotlin
package com.deepvoiceguard.app.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.compositionLocalOf

/**
 * 현재 적용된 brand 를 자식 Composable 들이 읽을 수 있도록 제공.
 * MainActivity 가 settings.carrierBrand → CarrierBrand 매핑 후 Theme 에 전달하면
 * 이 provider 를 통해 BrandHeader 등 자식 컴포넌트가 메타데이터 접근.
 */
val LocalCarrierBrand = compositionLocalOf { CarrierBrand.DEFAULT }

@Composable
fun DeepVoiceGuardTheme(
    brand: CarrierBrand = CarrierBrand.DEFAULT,
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    // brand-aware ColorScheme. Android 12+ dynamic color 는 brand override 와 충돌하므로 사용 안 함.
    val colorScheme = if (darkTheme) brand.darkScheme() else brand.lightScheme()

    CompositionLocalProvider(LocalCarrierBrand provides brand) {
        MaterialTheme(
            colorScheme = colorScheme,
            content = content,
        )
    }
}
```

- [ ] **Step 3.2: 빌드 + 커밋**

```bash
./gradlew :app:compileDebugKotlin 2>&1 | tail -3
```

기대: BUILD SUCCESSFUL.

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/ui/theme/Theme.kt
git commit -m "feat(theme): DeepVoiceGuardTheme accepts CarrierBrand + LocalCarrierBrand provider"
```

---

## Phase 4 — MainActivity 가 brand 구독

### Task 4: MainActivity 에서 settings.carrierBrand → Theme 전달

**Files:**
- Modify: `android-app/app/src/main/java/com/deepvoiceguard/app/ui/MainActivity.kt`

Architecture 결정대로 `remember { SettingsRepository(context) }` fallback 패턴을 사용. 이는 기존 `SettingsScreen.kt:38-42` 와 동일 패턴이며, top-level `preferencesDataStore` delegate (`SettingsRepository.kt:14`) 가 process-singleton 을 보장하므로 race 없음.

- [ ] **Step 4.1: 현재 MainActivity 구조 확인 (sanity check)**

```bash
grep -n "setContent\|DeepVoiceGuardTheme\|@AndroidEntryPoint" android-app/app/src/main/java/com/deepvoiceguard/app/ui/MainActivity.kt
```

기대: `setContent { DeepVoiceGuardTheme { DeepVoiceGuardApp() } }` 형태. `@AndroidEntryPoint` 어노테이션은 없을 가능성 높음 (확인용).

- [ ] **Step 4.2: setContent 안에서 settings 구독 + brand 매핑**

`MainActivity.kt` 상단 imports 에 추가:

```kotlin
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import com.deepvoiceguard.app.storage.AppSettings
import com.deepvoiceguard.app.storage.SettingsRepository
import com.deepvoiceguard.app.ui.theme.CarrierBrand
import com.deepvoiceguard.app.ui.theme.DeepVoiceGuardTheme
```

기존 `setContent { DeepVoiceGuardTheme { DeepVoiceGuardApp() } }` 를 다음으로 교체:

```kotlin
setContent {
    val context = LocalContext.current
    // SettingsRepository 는 DataStore wrapper. top-level preferencesDataStore delegate 가
    // process-singleton 을 보장 (SettingsRepository.kt:14), 인스턴스 다수 생성해도 동일 store.
    // 기존 SettingsScreen.kt:38-42 와 동일 패턴 — 검증됨. Hilt @Inject 변환은 후속 cleanup task.
    val repository = remember { SettingsRepository(context) }
    val settings by repository.settings.collectAsState(initial = AppSettings())
    val brand = remember(settings.carrierBrand) {
        CarrierBrand.fromKey(settings.carrierBrand)
    }
    DeepVoiceGuardTheme(brand = brand) {
        DeepVoiceGuardApp()
    }
}
```

- [ ] **Step 4.3: 빌드 + 커밋**

```bash
./gradlew :app:compileDebugKotlin 2>&1 | tail -3
```

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/ui/MainActivity.kt
git commit -m "feat(theme): MainActivity binds settings.carrierBrand to theme

- SettingsRepository wrapper 인스턴스 생성 (DataStore 는 process-singleton 이라 안전)
- carrierBrand 변경 시 collectAsState 가 recomposition 트리거 → Theme 즉시 갱신"
```

---

## Phase 5 — BrandHeader Composable

### Task 5: 상단 헤더 컴포넌트

**Files:**
- Create: `android-app/app/src/main/java/com/deepvoiceguard/app/ui/components/BrandHeader.kt`

- [ ] **Step 5.1: 컴포넌트 작성**

```kotlin
package com.deepvoiceguard.app.ui.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.deepvoiceguard.app.ui.theme.LocalCarrierBrand

/**
 * 화면 상단에 현재 brand 의 로고 + 이름을 표시.
 * Theme 의 LocalCarrierBrand 를 자동 구독.
 */
@Composable
fun BrandHeader(modifier: Modifier = Modifier) {
    val brand = LocalCarrierBrand.current
    Row(
        modifier = modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Image(
            painter = painterResource(id = brand.logoRes),
            contentDescription = brand.displayName,
            modifier = Modifier.size(36.dp),
        )
        Spacer(modifier = Modifier.width(12.dp))
        Text(
            text = brand.displayName,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )
    }
}
```

- [ ] **Step 5.2: HomeScreen 상단에 추가**

`HomeScreen.kt` 의 가장 바깥 Column 시작 직후에 `BrandHeader()` 호출 추가 + import:

```kotlin
import com.deepvoiceguard.app.ui.components.BrandHeader

// Column 안 가장 위에 추가 (기존 다른 카드 위)
BrandHeader()
```

- [ ] **Step 5.3: 빌드 + 커밋**

```bash
./gradlew :app:compileDebugKotlin 2>&1 | tail -3
```

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/ui/components/BrandHeader.kt \
         android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/HomeScreen.kt
git commit -m "feat(ui): BrandHeader on HomeScreen — auto-updates from LocalCarrierBrand"
```

---

## Phase 6 — Settings UI (RadioButton 그룹)

### Task 6: 브랜드 선택 카드

**Files:**
- Modify: `android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/SettingsScreen.kt`

상호배타 선택은 `RadioButton` 이 UX 표준. Switch 3개는 의미 모호.

- [ ] **Step 6.1: import 추가**

```kotlin
import androidx.compose.material3.RadioButton
import androidx.compose.foundation.clickable
import com.deepvoiceguard.app.ui.theme.CarrierBrand
```

- [ ] **Step 6.2: 카드 추가**

기존 SettingsScreen 안 적당한 위치에:

```kotlin
Spacer(modifier = Modifier.height(16.dp))

Card(modifier = Modifier.fillMaxWidth()) {
    Column(modifier = Modifier.padding(16.dp)) {
        Text(
            "브랜드 (데모용)",
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.Bold,
        )
        Spacer(modifier = Modifier.height(8.dp))
        for (b in CarrierBrand.values()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        scope.launch { repository.setCarrierBrand(b.key) }
                    }
                    .padding(vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                RadioButton(
                    selected = settings.carrierBrand == b.key,
                    onClick = { scope.launch { repository.setCarrierBrand(b.key) } },
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(b.displayName)
            }
        }
    }
}
```

- [ ] **Step 6.3: 빌드 + 커밋**

```bash
./gradlew :app:compileDebugKotlin 2>&1 | tail -3
```

```bash
git add android-app/app/src/main/java/com/deepvoiceguard/app/ui/screens/SettingsScreen.kt
git commit -m "feat(settings): brand selector — RadioButton group for mutual exclusion

3-way 상호배타 선택은 Switch 3개 보다 RadioButton 이 UX 표준."
```

---

## Phase 7 — 실기기 검증

### Task 7: 동작 확인

- [ ] **Step 7.1: ▶ Run**

- [ ] **Step 7.2: HomeScreen 상단에 기본 브랜드 헤더 (Anti-DeepVoice Guard 로고+이름) 표시 확인**

- [ ] **Step 7.3: Settings → 브랜드 → "Nova 모바일" 라디오 선택**

기대:
- HomeScreen 상단 로고 → 보라 별
- Header 텍스트 → "Nova 모바일"
- HomeScreen 의 Material primary color 사용 부분 (예: START 버튼) 보라 톤으로 변경
- **다른 화면 (Demo, History, Settings 본인, About) 의 BrandHeader 는 미적용** (HomeScreen 만 — 알려진 한계)

- [ ] **Step 7.4: "Lumi 모바일" 선택 → 주황색 + 원형 로고**

- [ ] **Step 7.5: 다시 "Anti-DeepVoice Guard" → 기본 복귀**

- [ ] **Step 7.6: 앱 재시작 후 마지막 선택 유지** (DataStore persistence 확인)

---

## Phase 8 — 백업 push

- [ ] **Step 8.1: 원격 push**

```bash
git push origin feature/brand-themes
```

---

## Self-Review Checklist

- [x] Spec 커버리지: 2개 가상 사업자 (Nova, Lumi) + 기본 = 3가지 테마. 즉시 반영. 데모 시연 가능.
- [x] Placeholder 없음.
- [x] 타입 일관성: `CarrierBrand.fromKey(nullable String)` ↔ `setCarrierBrand(brand.key)` 매칭.
- [x] LocalCarrierBrand provider 가 BrandHeader 와 다른 컴포넌트 모두에서 접근 가능.
- [x] 가상 브랜드명: 실존 통신사 (Mint, Aqua, KT, LG U+, SKT, KB, 7모바일 등) 와 충돌 없음. Nova/Lumi 는 일반 명사 조합으로 데모 안전.
- [x] Settings UI: 3-way 상호배타에 적합한 RadioButton 그룹 사용.

## 알려진 한계

- **BrandHeader 는 HomeScreen 에만** 적용. DemoScreen / HistoryScreen / SettingsScreen / AboutScreen 은 헤더 미적용 (시간 절약 + HomeScreen 이 데모 시 메인 화면이라 임팩트 충분). 후속 작업에서 모든 화면에 추가 가능.
- **앱 아이콘/스플래시(런처)** 는 변경 안 함 — Android build variant 가 필요. 데모 효과 대비 시간 비용 큼.
- **MainActivity 에서 SettingsRepository 직접 인스턴스화** (의도적 선택): top-level `preferencesDataStore` delegate 가 process-singleton 보장 — `SettingsScreen.kt:38-42` 가 같은 패턴으로 이미 사용 중이라 검증됨. Hilt `@AndroidEntryPoint` 변환은 후속 cleanup task — 본 plan 범위 밖.
- **Material 3 dynamic color 비활성화**: brand override 와 충돌하므로 의도적 — 사용자 시스템 색상은 brand 가 우선.
