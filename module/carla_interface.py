"""CARLA environment interface and lifecycle management."""

import math
import queue
import random
import time

import carla
import cv2
import numpy as np

from . import config as cfg


def is_allowed_npc_vehicle_blueprint(blueprint):
    """Return True for regular passenger-car NPC vehicle blueprints."""

    if not blueprint.has_attribute("number_of_wheels"):
        return False
    if int(blueprint.get_attribute("number_of_wheels")) != 4:
        return False

    blueprint_id = getattr(blueprint, "id", "").lower()
    return not any(keyword in blueprint_id for keyword in cfg.NPC_EXCLUDED_VEHICLE_KEYWORDS)


class CARLAInterface:
    """Interface for CARLA simulation."""

    def __init__(self):
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.sensors = {}
        self.sensor_queues = {}
        self.history_buffer = []
        self.npc_vehicle_ids = []
        self.npc_walker_ids = []
        self.npc_walker_controller_ids = []
        self.tm_port = 8000

        self.camera_configs = {
            "cam_front_left": {"x": 1.0, "y": -0.5, "z": 2.4, "pitch": 0.0, "yaw": -60.0, "fov": 120},
            "cam_front_wide": {"x": 1.5, "y": 0.0, "z": 2.4, "pitch": 0.0, "yaw": 0.0, "fov": 95},
            "cam_front_right": {"x": 1.0, "y": 0.5, "z": 2.4, "pitch": 0.0, "yaw": 60.0, "fov": 120},
            "cam_front_tele": {"x": 1.5, "y": 0.0, "z": 2.4, "pitch": 0.0, "yaw": 0.0, "fov": 30},
        }
        self.camera_order = ["cam_front_left", "cam_front_wide", "cam_front_right", "cam_front_tele"]

    def connect(self, host="localhost", port=2000):
        print(f"Connecting to CARLA at {host}:{port}...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        print("Connected to CARLA")

    def load_map(self, map_name):
        current_map = self.world.get_map().name
        if map_name not in current_map:
            print(f"Loading map: {map_name}...")
            self.world = self.client.load_world(map_name)
            print("Map load requested. Waiting for world tick...")
            self.world.wait_for_tick(20.0)
            time.sleep(1.0)
            print(f"Map loaded: {map_name}")
        else:
            print(f"Already on map: {current_map}")

    def enable_synchronous_mode(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(settings)
        tm = self.client.get_trafficmanager(self.tm_port)
        tm.set_synchronous_mode(True)
        print("Synchronous mode enabled.")

    def spawn_npcs(self, num_vehicles=cfg.NPC_VEHICLE_COUNT, num_walkers=cfg.NPC_WALKER_COUNT):
        bp_lib = self.world.get_blueprint_library()
        traffic_manager = self.client.get_trafficmanager(self.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        traffic_manager.global_percentage_speed_difference(0.0)

        vehicle_bps = [
            bp for bp in bp_lib.filter("vehicle.*") if is_allowed_npc_vehicle_blueprint(bp)
        ]
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        vehicle_count = min(num_vehicles, len(spawn_points))
        vehicle_batch = []
        for i in range(vehicle_count):
            bp = random.choice(vehicle_bps)
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "autopilot")
            transform = spawn_points[i]
            vehicle_batch.append(
                carla.command.SpawnActor(bp, transform).then(
                    carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port)
                )
            )
        vehicle_results = self.client.apply_batch_sync(vehicle_batch, True)
        for res in vehicle_results:
            if not res.error:
                self.npc_vehicle_ids.append(res.actor_id)
        print(f"Spawned NPC vehicles: {len(self.npc_vehicle_ids)}/{num_vehicles}")

        walker_bps = bp_lib.filter("walker.pedestrian.*")
        walker_spawn_points = []
        attempts = 0
        max_attempts = max(num_walkers * 5, 100)
        while len(walker_spawn_points) < num_walkers and attempts < max_attempts:
            loc = self.world.get_random_location_from_navigation()
            attempts += 1
            if loc is None:
                continue
            walker_spawn_points.append(carla.Transform(loc))

        walker_batch = []
        walker_speeds = []
        for transform in walker_spawn_points:
            bp = random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            speed = 1.4
            if bp.has_attribute("speed"):
                speed_values = bp.get_attribute("speed").recommended_values
                if len(speed_values) > 1:
                    speed = float(speed_values[1])
            walker_speeds.append(speed)
            walker_batch.append(carla.command.SpawnActor(bp, transform))

        walker_results = self.client.apply_batch_sync(walker_batch, True)
        spawned_walker_ids = []
        spawned_walker_speeds = []
        for idx, res in enumerate(walker_results):
            if not res.error:
                spawned_walker_ids.append(res.actor_id)
                spawned_walker_speeds.append(walker_speeds[idx])
        self.npc_walker_ids = spawned_walker_ids

        walker_controller_bp = bp_lib.find("controller.ai.walker")
        controller_batch = [
            carla.command.SpawnActor(walker_controller_bp, carla.Transform(), wid)
            for wid in self.npc_walker_ids
        ]
        controller_results = self.client.apply_batch_sync(controller_batch, True)
        self.npc_walker_controller_ids = [res.actor_id for res in controller_results if not res.error]
        controller_actors = self.world.get_actors(self.npc_walker_controller_ids)

        for i, controller in enumerate(controller_actors):
            controller.start()
            dest = self.world.get_random_location_from_navigation()
            if dest is not None:
                controller.go_to_location(dest)
            controller.set_max_speed(float(spawned_walker_speeds[i] if i < len(spawned_walker_speeds) else 1.4))

        print(f"Spawned NPC walkers: {len(self.npc_walker_ids)}/{num_walkers}")

    def spawn_ego_vehicle(self):
        print("Selecting spawn point...")
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute("role_name", "hero")
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        print("Selected random spawn point.")
        print("Spawning ego vehicle...")
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Spawned ego vehicle at {spawn_point.location}")
        return self.ego_vehicle

    def setup_cameras(self):
        print("Setting up cameras...")
        bp_lib = self.world.get_blueprint_library()
        for name, cfg_cam in self.camera_configs.items():
            print(f"  - spawning {name}")
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(cfg.IMG_WIDTH))
            cam_bp.set_attribute("image_size_y", str(cfg.IMG_HEIGHT))
            cam_bp.set_attribute("fov", str(cfg_cam["fov"]))
            cam_bp.set_attribute("enable_postprocess_effects", str(cfg.CAMERA_ENABLE_POSTPROCESS_EFFECTS))
            cam_bp.set_attribute("sensor_tick", "0.0")

            transform = carla.Transform(
                carla.Location(x=cfg_cam["x"], y=cfg_cam["y"], z=cfg_cam["z"]),
                carla.Rotation(pitch=cfg_cam["pitch"], yaw=cfg_cam["yaw"]),
            )
            sensor = self.world.spawn_actor(cam_bp, transform, attach_to=self.ego_vehicle)
            self.sensor_queues[name] = queue.Queue()
            sensor.listen(lambda data, n=name: self._camera_callback(data, n))
            self.sensors[name] = sensor
            time.sleep(0.2)

        print(f"Setup {len(self.sensors)} cameras ({cfg.IMG_WIDTH}x{cfg.IMG_HEIGHT})")

    def _camera_callback(self, image, name):
        if not self.sensor_queues[name].full():
            self.sensor_queues[name].put(image)

    def get_camera_images(self):
        images = []
        for name in self.camera_order:
            try:
                data = self.sensor_queues[name].get(timeout=1.0)
                array = np.frombuffer(data.raw_data, dtype=np.uint8)
                array = array.reshape((cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 4))[:, :, :3]
                array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                images.append(array)
            except queue.Empty:
                images.append(np.zeros((cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 3), dtype=np.uint8))
        return np.array(images)

    def get_ego_state(self):
        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        return {
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z,
            "roll": transform.rotation.roll,
            "pitch": transform.rotation.pitch,
            "yaw": transform.rotation.yaw,
            "speed": math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2),
        }

    def update_history(self, state):
        self.history_buffer.append(state)
        if len(self.history_buffer) > cfg.NUM_HISTORY:
            self.history_buffer.pop(0)

    def get_history_in_local_frame(self):
        if len(self.history_buffer) < cfg.NUM_HISTORY:
            while len(self.history_buffer) < cfg.NUM_HISTORY:
                self.history_buffer.insert(0, self.history_buffer[0] if self.history_buffer else self.get_ego_state())

        current = self.history_buffer[-1]
        current_pos = np.array([current["x"], current["y"], current["z"]])
        yaw_rad = math.radians(-current["yaw"])
        cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
        rot_matrix = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

        history_xyz = np.zeros((cfg.NUM_HISTORY, 3), dtype=np.float32)
        history_rot = np.zeros((cfg.NUM_HISTORY, 3, 3), dtype=np.float32)

        for i, st in enumerate(self.history_buffer):
            pos = np.array([st["x"], st["y"], st["z"]])
            history_xyz[i] = rot_matrix @ (pos - current_pos)
            state_yaw = math.radians(-st["yaw"])
            rel_yaw = state_yaw - yaw_rad
            history_rot[i] = np.array(
                [[math.cos(rel_yaw), -math.sin(rel_yaw), 0], [math.sin(rel_yaw), math.cos(rel_yaw), 0], [0, 0, 1]]
            )

        return history_xyz, history_rot

    def apply_control(self, steering, throttle, brake):
        control = carla.VehicleControl()
        control.steer = float(steering)
        control.throttle = float(throttle)
        control.brake = float(brake)
        self.ego_vehicle.apply_control(control)

    def tick(self):
        self.world.tick()

    def cleanup(self):
        print("\nCleaning up...")
        if self.client is None or self.world is None:
            return
        try:
            self.client.get_trafficmanager(self.tm_port).set_synchronous_mode(False)
        except Exception:
            pass
        if self.npc_walker_controller_ids:
            try:
                controllers = self.world.get_actors(self.npc_walker_controller_ids)
                for controller in controllers:
                    try:
                        controller.stop()
                    except Exception:
                        pass
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.npc_walker_controller_ids])
            except Exception:
                pass
        if self.npc_walker_ids:
            try:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.npc_walker_ids])
            except Exception:
                pass
        if self.npc_vehicle_ids:
            try:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.npc_vehicle_ids])
            except Exception:
                pass
        for sensor in self.sensors.values():
            try:
                sensor.stop()
                sensor.destroy()
            except Exception:
                pass
        if self.ego_vehicle:
            try:
                self.ego_vehicle.destroy()
            except Exception:
                pass
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except Exception:
            pass
