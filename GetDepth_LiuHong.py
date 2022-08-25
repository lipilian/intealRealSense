# %% import package
import pyrealsense2 as rs
import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import json
from matplotlib import cm
# %% configure depth and color camera
pipeline = rs.pipeline()
config = rs.config()
# %% Get device product
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
device_name = device.get_info(rs.camera_info.name)

print("The {} camera is connected to program\n".format(device_name))

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) # ! 1280 x720, 30fps depth image
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) # ! 1280 X 720, 30fps color image
jsonObj = json.load(open("cameraSettings.json"))
json_string = str(jsonObj).replace("'",'\"')

# ! Load external preset parameters about the camera
cfg = pipeline.start(config) # start streaming
device_running = cfg.get_device() # get running device information
advnc_mode = rs.rs400_advanced_mode(device_running) # get running configuration of device
advnc_mode.load_json(json_string)

# ! visualization method
colorizer = rs.colorizer()
histogram_equal_enable = colorizer.get_option(rs.option.histogram_equalization_enabled)
if histogram_equal_enable: # * check if histogram_equal_enable is enabled
    print("histogram equalization is enabled\n")
colorizer.set_option(rs.option.histogram_equalization_enabled, 1)
colorizer.set_option(rs.option.visual_preset, 0)
colorizer.set_option(rs.option.color_scheme, 0)

# ! declare filters
dec_filter = rs.decimation_filter(2.0)
dispar_filter = rs.disparity_transform(True)
spatial_filter = rs.spatial_filter(smooth_alpha = 0.5, smooth_delta = 20, magnitude = 2, hole_fill = 0)
temporal_filter = rs.temporal_filter(smooth_alpha = 0.4, smooth_delta = 20,persistence_control = 3)
depth_filter = rs.disparity_transform(False)
depth_scale = device_running.first_depth_sensor().get_depth_scale()


# %%
try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # * apply filters
        depth_frame = dec_filter.process(depth_frame)
        depth_frame = dispar_filter.process(depth_frame)
        depth_frame = spatial_filter.process(depth_frame)
        depth_frame = temporal_filter.process(depth_frame)
        depth_frame = depth_filter.process(depth_frame)
        #if not depth_frame or not color_frame:
        #    continue
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # !!!! ----------the data you want for depth image with 1280 x 720, 30fps-------------
        depth = depth_image * depth_scale 
        
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
            
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()

# %%

# %%
