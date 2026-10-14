import math
import random

NUM_DARTS = 15
CANONICAL_PX = 512.0
BOARD_METERS = 0.60
SCALE = BOARD_METERS / CANONICAL_PX

sdf_content = f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="dart_world">
    <gravity>0 0 0</gravity>
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Plugins -->
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics" />
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands" />
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster" />

    <!-- GUI Camera: Spawns directly facing the board at close range -->
    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>0 -1.4 1.2 0 0 1.570796</pose>
        <view_controller>orbit</view_controller>
      </camera>
    </gui>

    <!-- Bright Front Lighting -->
    <light type="directional" name="front_light">
      <cast_shadows>false</cast_shadows>
      <pose>0 -3 2 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>0 1 -0.2</direction>
    </light>

    <!-- Ground Floor -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal></plane></geometry>
        </collision>
      </link>
    </model>

    <!-- Dartboard Model Assembly -->
    <model name="dartboard">
      <pose>0 0 1.2 0 0 0</pose>

      <link name="carriage_x">
        <inertial><mass>0.5</mass><inertia><ixx>0.01</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.01</iyy><iyz>0</iyz><izz>0.01</izz></inertia></inertial>
      </link>
      <joint name="horizontal_joint" type="prismatic">
        <parent>world</parent><child>carriage_x</child>
        <axis><xyz>1 0 0</xyz><limit><lower>-1.0</lower><upper>1.0</upper><effort>1000000</effort></limit></axis>
      </joint>

      <link name="carriage_z">
        <inertial><mass>0.5</mass><inertia><ixx>0.01</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.01</iyy><iyz>0</iyz><izz>0.01</izz></inertia></inertial>
      </link>
      <joint name="vertical_joint" type="prismatic">
        <parent>carriage_x</parent><child>carriage_z</child>
        <axis><xyz>0 0 1</xyz><limit><lower>-1.0</lower><upper>1.0</upper><effort>1000000</effort></limit></axis>
      </joint>

      <link name="target_board">
        <pose>0 0 0 1.570796 0 0</pose>
        <inertial><mass>0.5</mass><inertia><ixx>0.02</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.02</iyy><iyz>0</iyz><izz>0.02</izz></inertia></inertial>
        <visual name="board_vis">
          <geometry><box><size>0.6 0.6 0.02</size></box></geometry>
          <material>
            <ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse>
            <pbr><metal><albedo_map>/home/jatin/dart_project/board.png</albedo_map></metal></pbr>
          </material>
        </visual>
        <collision name="board_col"><geometry><box><size>0.6 0.6 0.02</size></box></geometry></collision>
      </link>
      <joint name="board_anchor" type="fixed"><parent>carriage_z</parent><child>target_board</child></joint>

      <plugin filename="gz-sim-joint-controller-system" name="gz::sim::systems::JointController">
        <joint_name>horizontal_joint</joint_name><topic>/model/dartboard/joint/horizontal_joint/cmd_vel</topic>
      </plugin>
      <plugin filename="gz-sim-joint-controller-system" name="gz::sim::systems::JointController">
        <joint_name>vertical_joint</joint_name><topic>/model/dartboard/joint/vertical_joint/cmd_vel</topic>
      </plugin>
"""

print(f"[GENERATING] Rolling {NUM_DARTS} truly random polar darts:")

for i in range(1, NUM_DARTS + 1):
    # True uniform circular distribution (prevents bullseye clustering bias)
    r = 200.0 * math.sqrt(random.uniform(0.0, 1.0))
    theta = random.uniform(0.0, 2.0 * math.pi)

    px_x = 256.0 + r * math.cos(theta)
    px_y = 256.0 + r * math.sin(theta)

    rel_x = px_x - 256.0
    rel_y = px_y - 256.0
    dist = math.hypot(rel_x, rel_y)

    xm = rel_x * SCALE
    ym = -rel_y * SCALE

    print(
        f"  Dart #{i:02d}: Screen=({px_x:.1f}, {px_y:.1f}) | r={dist:5.1f}px | "
        f"3D=({xm:+.4f}m, {ym:+.4f}m)"
    )

    sdf_content += f"""
      <link name="dart_{i}_link">
        <pose relative_to="target_board">{xm:.5f} {ym:.5f} -0.08 0 0 0</pose>
        <inertial><mass>0.01</mass><inertia><ixx>1e-5</ixx><ixy>0</ixy><ixz>0</ixz><iyy>1e-5</iyy><iyz>0</iyz><izz>1e-5</izz></inertia></inertial>
        <visual name="v{i}">
          <geometry><cylinder><radius>0.016</radius><length>0.03</length></cylinder></geometry>
          <material>
            <ambient>0 1 0 1</ambient><diffuse>0 1 0 1</diffuse><emissive>0 0.9 0 1</emissive>
          </material>
        </visual>
      </link>
      <joint name="dart_{i}_joint" type="prismatic">
        <parent>target_board</parent><child>dart_{i}_link</child>
        <axis><xyz expressed_in="target_board">0 0 1</xyz><limit><lower>0.0</lower><upper>0.10</upper><effort>1000</effort></limit></axis>
      </joint>
      <plugin filename="gz-sim-joint-controller-system" name="gz::sim::systems::JointController">
        <joint_name>dart_{i}_joint</joint_name><topic>/model/dartboard/joint/dart_{i}_joint/cmd_vel</topic>
      </plugin>
"""

sdf_content += """
    </model>
  </world>
</sdf>
"""

with open("dartboard_world.sdf", "w") as f:
    f.write(sdf_content)

print("[OK] 'dartboard_world.sdf' regenerated with uniform polar randomization.\n")