
import pyaudio
import pvporcupine
import os
from dotenv import load_dotenv

def test():
    print("--- AUDIO DIAGNOSIS ---")
    
    # 1. Check Env
    appdata = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi")
    env_path = os.path.join(appdata, ".env")
    load_dotenv(env_path)
    
    access_key = os.getenv("PVPORCUPINE_PRIVATE_KEY")
    print(f"Porcupine Key present: {bool(access_key)}")
    
    # 2. List Audio Devices
    try:
        pa = pyaudio.PyAudio()
        print("\nInput Devices:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get('maxInputChannels') > 0:
                print(f"  [{i}] {info.get('name')}")
        pa.terminate()
    except Exception as e:
        print(f"PyAudio error: {e}")

    # 3. Test Porcupine Init
    if access_key:
        try:
            handle = pvporcupine.create(access_key=access_key, keywords=["computer"])
            print("\nPorcupine initialized successfully with 'computer'.")
            handle.delete()
        except Exception as e:
            print(f"Porcupine init failed: {e}")
            
    # 4. Check PPN file
    ppn = r"C:\Users\suraj\Desktop\Ai_Assistant\SentinelAi\assets\sounds\Hey-robert_en_windows_v3_0_0.ppn"
    if os.path.exists(ppn):
        print(f"\nCustom PPN exists: {ppn}")
        if access_key:
            try:
                handle = pvporcupine.create(access_key=access_key, keyword_paths=[ppn])
                print("Custom PPN initialized successfully!")
                handle.delete()
            except Exception as e:
                print(f"Custom PPN init failed: {e}")
    else:
        print(f"\nCustom PPN NOT found at {ppn}")

if __name__ == "__main__":
    test()
