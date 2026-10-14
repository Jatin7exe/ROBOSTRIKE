import math
import time
import subprocess

def main():
    print("[INFO] Starting Corrected Linear To-and-Fro Motion Controller")
    print("[INFO] Publishing velocity via gz.msgs.Double...")

    # Parameters for harmonic to-and-fro motion along X-axis
    amplitude = 0.45      # Maximum horizontal displacement in meters (+/- 0.45m)
    frequency = 0.4       # Hz (complete back-and-forth cycle every 2.5 seconds)
    
    # Dart actuation timing parameters
    num_darts = 15
    deploy_interval = 5.0  # Deploy a dart every 5 seconds
    start_time = time.time()
    next_dart_index = 1

    try:
        while True:
            t = time.time() - start_time
            
            # Harmonic motion equation:
            # Position x(t) = Amplitude * sin(omega * t)
            # Velocity v(t) = dx/dt = Amplitude * omega * cos(omega * t)
            omega = 2.0 * math.pi * frequency
            v_x = amplitude * omega * math.cos(omega * t)
            
            # CORRECTED: Gazebo JointController expects gz.msgs.Double with payload "data: <float>"
            cmd_vel_str = (
                f"gz topic -t /model/dartboard/joint/horizontal_joint/cmd_vel "
                f"-m gz.msgs.Double -p \"data: {v_x}\""
            )
            subprocess.run(cmd_vel_str, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Sequentially deploy hidden darts every `deploy_interval` seconds
            elapsed = time.time() - start_time
            expected_dart = int(elapsed // deploy_interval) + 1
            
            if expected_dart > next_dart_index and next_dart_index <= num_darts:
                joint_name = f"dart_{next_dart_index}_joint"
                dart_cmd = (
                    f"gz topic -t /model/dartboard/joint/{joint_name}/cmd_vel "
                    f"-m gz.msgs.Double -p \"data: 0.05\""
                )
                subprocess.run(dart_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[ACTION] Deployed Dart #{next_dart_index:02d}")
                next_dart_index += 1

            time.sleep(0.02) # ~50 Hz control loop

    except KeyboardInterrupt:
        print("\n[INFO] Linear motion controller stopped by user.")

if __name__ == "__main__":
    main()