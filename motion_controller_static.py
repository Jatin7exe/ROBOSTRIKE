import time
import subprocess

def main():
    print("[INFO] Starting Static Dartboard Controller (Board Stationary)")
    print("[INFO] Sequentially deploying darts onto the non-moving target...")

    num_darts = 15
    deploy_interval = 5.0  # Deploy a dart every 5 seconds
    start_time = time.time()
    next_dart_index = 1

    try:
        while next_dart_index <= num_darts:
            elapsed = time.time() - start_time
            expected_dart = int(elapsed // deploy_interval) + 1

            if expected_dart > next_dart_index and next_dart_index <= num_darts:
                joint_name = f"dart_{next_dart_index}_joint"
                # Push the hidden dart forward onto the stationary board face (+0.05m velocity push)
                dart_cmd = (
                    f"gz topic -t /model/dartboard/joint/{joint_name}/cmd_vel "
                    f"-m gz.msgs.Double -p \"data: 0.05\""
                )
                subprocess.run(dart_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[ACTION] Deployed Dart #{next_dart_index:02d} onto static board")
                next_dart_index += 1

            time.sleep(0.1)

        print("[INFO] All darts deployed. Keeping controller alive...")
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[INFO] Static controller stopped by user.")

if __name__ == "__main__":
    main()