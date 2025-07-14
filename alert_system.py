import winsound
import os
import sys

def send_alert(name):
    try:
        frequency = 2500  # Frequency in Hertz
        duration = 1000   # Duration in milliseconds (1 second)
        winsound.Beep(frequency, duration)  # Beep alert

        if sys.platform == "darwin" or sys.platform.startswith("linux"):
            os.system('say "Unknown person detected"')  # macOS/Linux text-to-speech
        else:
            os.system('echo "Unknown person detected"')  # Windows alternative

        print(f" ALERT: Unknown person '{name}' detected! ")

    except KeyboardInterrupt:
        print("\n[INFO] Alert system interrupted safely. Exiting...")
    except Exception as e:
        print(f"[ERROR] Failed to send alert: {e}")

