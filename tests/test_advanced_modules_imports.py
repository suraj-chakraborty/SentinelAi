"""
tests/test_advanced_modules_imports.py
──────────────────────────────────────
Test all advanced module imports.
"""

import sys
import os
import importlib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

modules_to_test = [
    ('sentinel.memory.screen_capture', 'ScreenCapture'),
    ('sentinel.memory.screen_memory_db', 'ScreenMemoryDB'),
    ('sentinel.memory.screen_indexer', 'ScreenIndexer'),
    ('sentinel.commands.search_memory', 'SearchMemoryCommand'),
    ('sentinel.core.plugin_generator', 'PluginGenerator'),
    ('sentinel.core.dynamic_plugin_loader', 'DynamicPluginLoader'),
    ('sentinel.automation.uia_automation', 'UIAutomation'),
    ('sentinel.commands.ui_automation', 'UIAutomationCommand'),
    ('sentinel.vision.live_vision', 'LiveVisionStream'),
    ('sentinel.commands.live_vision', 'LiveVisionCommand'),
    ('sentinel.daemons.daemon_manager', 'DaemonManager'),
    ('sentinel.commands.daemon_control', 'DaemonControlCommand'),
    ('sentinel.voice.acoustic_awareness', 'AcousticAwareness'),
    ('sentinel.core.sandbox', 'SandboxEngine'),
    ('sentinel.perception.biometric_auth', 'BiometricAuth'),
    ('sentinel.core.phone_bridge', 'PhoneBridge'),
    ('sentinel.voice.meeting_surrogate', 'MeetingSurrogate'),
    ('sentinel.security.overwatch', 'CyberOverwatch'),
    ('sentinel.network.hive_computing', 'HiveNetwork'),
    ('sentinel.core.nocturnal_trainer', 'NocturnalTrainer'),
    ('sentinel.core.prefetch_engine', 'NeuralPrefetch'),
    ('sentinel.ar.spatial_interface', 'SpatialInterface'),
    ('sentinel.audio.omni_ear', 'OmniEar'),
    ('sentinel.voice.emotional_tts', 'EmotionalTTS'),
    ('sentinel.neuro.telekinesis', 'Telekinesis'),
]

print('Module Import Test Results:')
print('=' * 60)
passed = 0
failed = 0

for module_path, class_name in modules_to_test:
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name, None)
        if cls:
            print(f'[PASS] {module_path}.{class_name}')
            passed += 1
        else:
            print(f'[FAIL] {module_path}.{class_name} - class not found')
            failed += 1
    except Exception as e:
        print(f'[FAIL] {module_path}.{class_name} - {type(e).__name__}: {e}')
        failed += 1

print('=' * 60)
print(f'Total: {passed} passed, {failed} failed')