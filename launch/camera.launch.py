from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Launch file to start the usb_cam node and then throttle its stream.
    """

    # --- 1. USB Camera Node ---
    # https://github.com/ros-drivers/usb_cam
    # The usb_cam node will publish its raw image stream to /camera/image_raw
    camera_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        namespace='camera',
        parameters=[
            {'pixel_format': 'yuyv'},
            {'io_method': 'mmap'},
            {'framerate': 15.0}, # You can set a higher framerate here if desired
            {'image_width': 640}, # Example: Add specific resolution
            {'image_height': 480}, # Example: Add specific resolution
        ],
    )

    # --- 2. Topic Tools Throttler Node ---
    # https://github.com/ros-tooling/topic_tools
    # This node will subscribe to the /camera/image_raw topic and
    # publish a throttled version to /camera/image_raw/throttled at 1 Hz.
    throttler_node = Node(
        package='topic_tools',
        executable='throttle',
        name='image_throttler',
        namespace='camera',
        arguments=[
            'messages',                   # Throttle by messages
            '/camera/image_raw',          # Input topic (full topic name after namespace)
            '0.5',                        # Throttle rate: 1.0 messages per second (1 Hz)
            '/camera/image_raw/throttled' # Output topic (full topic name after namespace)
            
        ],
    )

    return LaunchDescription([
        camera_node,
        throttler_node
    ])