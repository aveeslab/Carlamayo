import glob
import os
import sys
import queue
import numpy as np
import cv2
import carla
import random
import time
import json
import logging

# ==============================================================================
# 1. 설정 (Configuration)
# ==============================================================================
OUTPUT_DIR = "carla_data"
# 원래 해상도로 복구 (그래픽 문제 아님)
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
FPS = 10
FIXED_DELTA_SECONDS = 1.0 / FPS  # 0.1s
WARMUP_SECONDS = 2.0             # 2초간 데이터 수집 없이 주행

# NPC 설정
NUM_NPC_VEHICLES = 20
NUM_NPC_WALKERS = 30

# 센서 위치
SENSOR_CONFIGS = {
    "cam_front_wide": {"x": 1.5, "y": 0.0, "z": 2.4, "pitch": 0.0, "yaw": 0.0, "fov": 120},
    "cam_front_tele": {"x": 1.5, "y": 0.0, "z": 2.4, "pitch": 0.0, "yaw": 0.0, "fov": 30},
    "cam_front_left": {"x": 1.0, "y": -0.5, "z": 2.4, "pitch": 0.0, "yaw": -60.0, "fov": 120},
    "cam_front_right": {"x": 1.0, "y": 0.5, "z": 2.4, "pitch": 0.0, "yaw": 60.0, "fov": 120},
    "cam_rear_left": {"x": -0.5, "y": -0.5, "z": 2.4, "pitch": 0.0, "yaw": -120.0, "fov": 120},
    "cam_rear_right": {"x": -0.5, "y": 0.5, "z": 2.4, "pitch": 0.0, "yaw": 120.0, "fov": 120},
    "cam_rear_wide": {"x": -1.5, "y": 0.0, "z": 2.4, "pitch": 0.0, "yaw": 180.0, "fov": 120},
    "cam_rear_tele": {"x": -1.5, "y": 0.0, "z": 2.4, "pitch": 0.0, "yaw": 180.0, "fov": 30},
}


def _frame_file_path(output_dir, sensor_name, frame_id):
    ext = "ply" if "lidar" in sensor_name else "jpg"
    return os.path.join(output_dir, sensor_name, f"{frame_id:06d}.{ext}")


def _frame_is_complete(output_dir, sensor_names, frame_id):
    for sensor_name in sensor_names:
        if not os.path.exists(_frame_file_path(output_dir, sensor_name, frame_id)):
            return False
    return True

# ==============================================================================
# 2. 유틸리티 함수
# ==============================================================================
def sensor_callback(sensor_data, sensor_queue, sensor_name):
    # 동기 모드에서는 틱 타이밍이 중요하므로 큐에 넣을 때 블로킹 없이 넣음
    if not sensor_queue.full():
        sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))

def spawn_npc(client, world, tm_port, num_vehicles, num_walkers):
    print(f"Spawning {num_vehicles} vehicles and {num_walkers} walkers...")
    actor_list = []

    # --- NPC 차량 ---
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter("vehicle.*")
    vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]
    spawn_points = world.get_map().get_spawn_points()

    number_of_spawn_points = len(spawn_points)
    if num_vehicles < number_of_spawn_points:
        random.shuffle(spawn_points)
    else:
        num_vehicles = number_of_spawn_points

    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= num_vehicles: break
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        bp.set_attribute('role_name', 'autopilot')
        batch.append(carla.command.SpawnActor(bp, transform)
            .then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port)))

    results = client.apply_batch_sync(batch, True)
    for response in results:
        if not response.error:
            actor_list.append(world.get_actor(response.actor_id))
    
    # --- NPC 보행자 ---
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    walker_controller_bp = bp_lib.find('controller.ai.walker')
    
    spawn_points = []
    for _ in range(num_walkers):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point.location = loc
            spawn_points.append(spawn_point)

    batch = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(walker_bps)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
    
    results = client.apply_batch_sync(batch, True)
    walkers_list = []
    for response in results:
        if not response.error:
            walkers_list.append(world.get_actor(response.actor_id))
            actor_list.append(world.get_actor(response.actor_id))

    batch = []
    for walker in walkers_list:
        batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walker))
    
    results = client.apply_batch_sync(batch, True)
    for response in results:
        if not response.error:
            controller = world.get_actor(response.actor_id)
            actor_list.append(controller)
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(1.4 + random.random())

    return actor_list

# ==============================================================================
# 3. 메인 실행 루프
# ==============================================================================
def main():
    actor_list = []
    trajectory_data = {}
    expected_sensor_names = list(SENSOR_CONFIGS.keys()) + ["lidar_top"]

    try:
        # 클라이언트 연결
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.load_world('Town02')
        bp_lib = world.get_blueprint_library()

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        for name in SENSOR_CONFIGS.keys():
            os.makedirs(os.path.join(OUTPUT_DIR, name), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "lidar_top"), exist_ok=True)

        # Settings
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)

        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        tm.set_global_distance_to_leading_vehicle(2.5)

        # Actors
        npc_actors = spawn_npc(client, world, tm.get_port(), NUM_NPC_VEHICLES, NUM_NPC_WALKERS)
        actor_list.extend(npc_actors)

        # Ego Vehicle
        spawn_points = world.get_map().get_spawn_points()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        vehicle_bp.set_attribute('role_name', 'hero')

        ego_vehicle = None
        for _ in range(10):
            start_pose = random.choice(spawn_points)
            ego_vehicle = world.spawn_actor(vehicle_bp, start_pose)
            if ego_vehicle is not None: break

        if ego_vehicle is None:
            print("Error: Could not spawn ego vehicle.")
            return

        actor_list.append(ego_vehicle)
        ego_vehicle.set_autopilot(True, tm.get_port())

        # Sensors
        sensor_queue = queue.Queue()

        for name, cfg in SENSOR_CONFIGS.items():
            cam_bp = bp_lib.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
            cam_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
            cam_bp.set_attribute('fov', str(cfg['fov']))
            cam_bp.set_attribute('sensor_tick', '0.0')

            transform = carla.Transform(
                carla.Location(x=cfg['x'], y=cfg['y'], z=cfg['z']),
                carla.Rotation(pitch=cfg['pitch'], yaw=cfg['yaw'])
            )

            sensor = world.spawn_actor(cam_bp, transform, attach_to=ego_vehicle)
            sensor.listen(lambda data, n=name: sensor_callback(data, sensor_queue, n))
            actor_list.append(sensor)

        # LiDAR
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('rotation_frequency', str(FPS))
        lidar_bp.set_attribute('points_per_second', '1200000')
        lidar_bp.set_attribute('sensor_tick', '0.0')

        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
        lidar_sensor.listen(lambda data: sensor_callback(data, sensor_queue, "lidar_top"))
        actor_list.append(lidar_sensor)

        total_sensors = len(SENSOR_CONFIGS) + 1

        # Warm-up
        warmup_frames = int(WARMUP_SECONDS * FPS)
        print(f"Warm-up: Ticking world {warmup_frames} times (ignoring queue)...")

        for _ in range(warmup_frames):
            world.tick()

        print("Clearing initial queue garbage...")
        with sensor_queue.mutex:
            sensor_queue.queue.clear()

        print("Start recording loop!")

        # Main Loop
        while True:
            world.tick()
            tf = ego_vehicle.get_transform()

            current_frame_data = {}
            try:
                for _ in range(total_sensors):
                    frame, name, data = sensor_queue.get(True, 5.0)
                    current_frame_data[name] = data
            except queue.Empty:
                print("Warning: Sensor data missing (Timeout). Skipping frame.")
                continue

            if len(current_frame_data) != total_sensors:
                print("Warning: Incomplete frame data. Skipping.")
                continue

            frame_id = list(current_frame_data.values())[0].frame

            print(f"Recording Frame: {frame_id}")

            frame_write_ok = True
            for name in expected_sensor_names:
                data = current_frame_data[name]
                if "lidar" in name:
                    data.save_to_disk(_frame_file_path(OUTPUT_DIR, name, frame_id))
                else:
                    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (data.height, data.width, 4))
                    array = array[:, :, :3]
                    filename = _frame_file_path(OUTPUT_DIR, name, frame_id)
                    if not cv2.imwrite(filename, array):
                        print(f"Warning: Failed to write image for {name} frame {frame_id}.")
                        frame_write_ok = False

            # Record pose only after all sensor files are confirmed on disk.
            if frame_write_ok and _frame_is_complete(OUTPUT_DIR, expected_sensor_names, frame_id):
                trajectory_data[frame_id] = {
                    "x": round(tf.location.x, 6),
                    "y": round(tf.location.y, 6),
                    "z": round(tf.location.z, 6),
                    "pitch": round(tf.rotation.pitch, 6),
                    "yaw": round(tf.rotation.yaw, 6),
                    "roll": round(tf.rotation.roll, 6)
                }
            else:
                print(f"Warning: Incomplete frame {frame_id}; excluded from trajectory.json.")

    except KeyboardInterrupt:
        print("\nStopping recording...")

    finally:
        if trajectory_data:
            # Final safety pass: drop any trajectory frames missing sensor files.
            filtered_trajectory_data = {}
            dropped = 0
            for frame_id, pose in trajectory_data.items():
                if _frame_is_complete(OUTPUT_DIR, expected_sensor_names, int(frame_id)):
                    filtered_trajectory_data[frame_id] = pose
                else:
                    dropped += 1

            if dropped > 0:
                print(f"Dropped {dropped} incomplete frame(s) from trajectory.json.")

            json_path = os.path.join(OUTPUT_DIR, 'trajectory.json')
            with open(json_path, 'w') as f:
                json.dump(filtered_trajectory_data, f, indent=4)

        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)

        print("Destroying actors...")
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print("Done.")

if __name__ == '__main__':
    main()
