import math
import subprocess
import threading
import time

# Motion Parameters
ORBIT_RADIUS = 0.35  # meters
ORBIT_PERIOD = 5.0  # seconds per circular loop
OMEGA = 2.0 * math.pi / ORBIT_PERIOD
SPAWN_INTERVAL = 5.0  # seconds between throws

HORIZONTAL_TOPIC = "/model/dartboard/joint/horizontal_joint/cmd_vel"
VERTICAL_TOPIC = "/model/dartboard/joint/vertical_joint/cmd_vel"

running = True


def count_available_darts():
  """Counts how many darts are defined in dartboard_world.sdf."""
  count = 0
  try:
    with open("dartboard_world.sdf", "r") as f:
      text = f.read()
      while f"dart_{count + 1}_joint" in text:
        count += 1
  except Exception:
    count = 10
  return count


TOTAL_DARTS = count_available_darts()


def set_velocity(topic, velocity):
  cmd = [
      "gz",
      "topic",
      "-t",
      topic,
      "-m",
      "gz.msgs.Double",
      "-p",
      f"data: {velocity:.5f}",
  ]
  subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def motion_loop():
  """Generates orbital revolution along X-Z plane."""
  t_start = time.time()
  while running:
    elapsed = time.time() - t_start
    vx = ORBIT_RADIUS * OMEGA * math.cos(OMEGA * elapsed)
    vz = -ORBIT_RADIUS * OMEGA * math.sin(OMEGA * elapsed)

    set_velocity(HORIZONTAL_TOPIC, vx)
    set_velocity(VERTICAL_TOPIC, vz)
    time.sleep(0.05)


def dart_sequence_loop():
  """Deploys each random dart from behind the board onto the front face."""
  time.sleep(3.0)  # Wait for initial revolution stabilization

  for dart_num in range(1, TOTAL_DARTS + 1):
    if not running:
      break

    joint_topic = f"/model/dartboard/joint/dart_{dart_num}_joint/cmd_vel"
    # Drive joint forward to upper limit (+0.02m) onto front face
    set_velocity(joint_topic, 0.5)
    time.sleep(0.25)
    set_velocity(joint_topic, 0.0)

    print(
        f"🎯 [THROWN] Random Dart #{dart_num}/{TOTAL_DARTS} anchored to the"
        " revolving target!"
    )
    time.sleep(SPAWN_INTERVAL)

  print(
      "\n[INFO] All darts deployed. Target continues revolving indefinitely."
  )


print("==================================================")
print("  REVOLVING DART SIMULATION CONTROLLER")
print(f"  • Orbital Period:        {ORBIT_PERIOD} s")
print(f"  • Total Random Darts:    {TOTAL_DARTS}")
print(f"  • Throw Interval:        {SPAWN_INTERVAL} s")
print("  Press Ctrl+C to halt.")
print("==================================================\n")

t_motion = threading.Thread(target=motion_loop, daemon=True)
t_darts = threading.Thread(target=dart_sequence_loop, daemon=True)

t_motion.start()
t_darts.start()

try:
  while True:
    time.sleep(1.0)
except KeyboardInterrupt:
  print("\nStopping controller...")
  running = False
  set_velocity(HORIZONTAL_TOPIC, 0.0)
  set_velocity(VERTICAL_TOPIC, 0.0)
  print("Halted.")