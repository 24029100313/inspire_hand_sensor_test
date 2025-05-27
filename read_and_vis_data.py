from pydoc import cli
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# Updated imports for newer pymodbus versions
import asyncio
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

# Define Modbus TCP parameters
MODBUS_IP = "192.168.11.210"
MODBUS_PORT = 6000

# Define address ranges for each finger/part
TOUCH_SENSOR_BASE_ADDR_PINKY = 3000  # Pinky finger
TOUCH_SENSOR_END_ADDR_PINKY = 3369

TOUCH_SENSOR_BASE_ADDR_RING = 3370  # Ring finger
TOUCH_SENSOR_END_ADDR_RING = 3739

TOUCH_SENSOR_BASE_ADDR_MIDDLE = 3740  # Middle finger
TOUCH_SENSOR_END_ADDR_MIDDLE = 4109

TOUCH_SENSOR_BASE_ADDR_INDEX = 4110  # Index finger
TOUCH_SENSOR_END_ADDR_INDEX = 4479

TOUCH_SENSOR_BASE_ADDR_THUMB = 4480  # Thumb
TOUCH_SENSOR_END_ADDR_THUMB = 4899

TOUCH_SENSOR_BASE_ADDR_PALM = 4900  # Palm
TOUCH_SENSOR_END_ADDR_PALM = 5123

# Modbus maximum registers per read
MAX_REGISTERS_PER_READ = 125

# Define finger dimensions and their segments based on documentation
# Each finger has 3 parts: tip (指端), pad (指尖), and base (指腹)
# Thumb has 4 parts: tip (指端), pad (指尖), middle (指中), and base (指腹)
FINGER_SEGMENTS = {
    "pinky": [
        {"name": "tip", "dims": (3, 3), "addr_range": (3000, 3017)},  # 指端
        {"name": "pad", "dims": (12, 8), "addr_range": (3018, 3209)},  # 指尖
        {"name": "base", "dims": (10, 8), "addr_range": (3210, 3369)},  # 指腹
    ],
    "ring": [
        {"name": "tip", "dims": (3, 3), "addr_range": (3370, 3387)},  # 指端
        {"name": "pad", "dims": (12, 8), "addr_range": (3388, 3579)},  # 指尖
        {"name": "base", "dims": (10, 8), "addr_range": (3580, 3739)},  # 指腹
    ],
    "middle": [
        {"name": "tip", "dims": (3, 3), "addr_range": (3740, 3757)},  # 指端
        {"name": "pad", "dims": (12, 8), "addr_range": (3758, 3949)},  # 指尖
        {"name": "base", "dims": (10, 8), "addr_range": (3950, 4109)},  # 指腹
    ],
    "index": [
        {"name": "tip", "dims": (3, 3), "addr_range": (4110, 4127)},  # 指端
        {"name": "pad", "dims": (12, 8), "addr_range": (4128, 4319)},  # 指尖
        {"name": "base", "dims": (10, 8), "addr_range": (4320, 4479)},  # 指腹
    ],
    "thumb": [
        {"name": "tip", "dims": (3, 3), "addr_range": (4480, 4497)},  # 指端
        {"name": "pad", "dims": (12, 8), "addr_range": (4498, 4689)},  # 指尖
        {"name": "middle", "dims": (3, 3), "addr_range": (4690, 4707)},  # 指中
        {"name": "base", "dims": (12, 8), "addr_range": (4708, 4899)},  # 指腹
    ],
    "palm": [
        {"name": "palm", "dims": (14, 8), "addr_range": (4900, 5123)}  # 掌心
    ],
}

# Define dimensions for the overall visualization
FINGER_DIMENSIONS = {
    "pinky": (25, 8),  # Combined height x width
    "ring": (25, 8),
    "middle": (25, 8),
    "index": (25, 8),
    "thumb": (30, 8),
    "palm": (14, 8),
}

# Define the sensor layout based on the image shown
FINGER_LABELS = {
    "pinky": "pinky",
    "ring": "ring",
    "middle": "middle",
    "index": "index",
    "thumb": "thumb",
    "palm": "palm",
}

# Define visualization layout to better match the hand anatomy
HAND_LAYOUT = [
    (0, 0, "thumb"),  # Left side, top row
    (0, 1, "index"),  # 2nd position
    (0, 2, "middle"),  # 3rd position
    (0, 3, "ring"),  # 4th position
    (0, 4, "pinky"),  # Right side
    (1, 1, "palm"),  # Bottom center
]


def read_register_range(client, start_addr, end_addr):
    """
    Read registers in batches within the specified address range.
    """
    register_values = []

    for addr in range(start_addr, end_addr, MAX_REGISTERS_PER_READ * 2):
        current_count = min(MAX_REGISTERS_PER_READ, (end_addr - addr) // 2 + 1)

        try:
            # Try the new API format first
            try:
                response = client.read_holding_registers(address=addr, count=current_count)
            except TypeError:
                # Fall back to old API format
                response = client.read_holding_registers(addr, current_count)
                
            # Extract registers from the response
            if hasattr(response, 'registers'):
                register_values.extend(response.registers)
            elif isinstance(response, list):
                register_values.extend(response)
            else:
                print(f"Unexpected response format: {type(response)}")
                register_values.extend([0] * current_count)
                
        except ModbusException as e:
            print(f"Failed to read register {addr}: {e}")
            register_values.extend([0] * current_count)
        except Exception as e:
            print(f"Unexpected error reading register {addr}: {e}")
            register_values.extend([0] * current_count)

    return register_values


def reshape_data(data, dimensions):
    """
    Reshape the 1D data into a 2D array based on specified dimensions.
    """
    height, width = dimensions
    reshaped_data = np.zeros((height, width))

    # Ensure data length matches the required size
    expected_size = height * width
    data = data[:expected_size]

    # Pad with zeros if necessary
    if len(data) < expected_size:
        data = data + [0] * (expected_size - len(data))

    # Reshape data into 2D
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            if idx < len(data):
                reshaped_data[i, j] = data[idx]

    return reshaped_data


def read_segment_data(client, segment, use_simulated_data=False, frame=0):
    """
    Read data for a specific finger segment.
    If simulated data is used, generate a pattern that resembles touch data.
    """
    start_addr, end_addr = segment["addr_range"]
    height, width = segment["dims"]
    expected_size = height * width

    if use_simulated_data:
        # Generate simulated data
        data = []
        for i in range(expected_size):
            # Create different patterns for different segments
            if segment["name"] == "tip":
                # Tips have higher values in the center
                center_dist = (
                    (i % width - width / 2) ** 2 + (i // width - height / 2) ** 2
                ) ** 0.5
                val = int(
                    800 * np.exp(-center_dist) * (0.7 + 0.3 * np.sin(frame * 0.1))
                )
            elif segment["name"] == "pad":
                # Pads have wave patterns
                val = int(
                    500
                    + 400
                    * np.sin(i * 0.2 + frame * 0.1)
                    * np.cos(i * 0.1 + frame * 0.08)
                )
            elif segment["name"] == "middle":
                # Middle section has circular patterns
                center_dist = (
                    (i % width - width / 2) ** 2 + (i // width - height / 2) ** 2
                ) ** 0.5
                val = int(
                    600 * np.exp(-center_dist / 2) * (0.8 + 0.2 * np.cos(frame * 0.15))
                )
            else:  # base or palm
                # Base and palm have random-like patterns
                val = int(
                    300
                    + 200
                    * np.sin(i * 0.3 + frame * 0.05)
                    * np.sin(i * 0.2 + frame * 0.07)
                )

            # Ensure value is in valid range
            val = max(0, min(1023, val))
            data.append(val)

        return data
    else:
        # Read real data from Modbus
        try:
            return read_register_range(client, start_addr, end_addr)
        except Exception as e:
            print(f"Error reading segment {segment['name']}: {e}")
            return [0] * expected_size

class TouchSensorReader:
    def __init__(self, client=None, use_simulated_data=False):
        """
        Initialize the TouchSensorReader.
        
        Args:
            use_simulated_data: If True, use simulated data instead of reading from the device.
        """
        self.use_simulated_data = use_simulated_data
        self.client = client
        self.frame = 0
        
        if not self.use_simulated_data:
            self._connect()
    
    def _connect(self):
        """Establish connection to the Modbus device."""
        if self.client is None:
            try:
                self.client = ModbusTcpClient(host=MODBUS_IP, port=MODBUS_PORT)
                self.client.connect()
                print(f"[Touch INFO] Successfully connected to Modbus device at {MODBUS_IP}:{MODBUS_PORT}")
            except Exception as e:
                print(f"[Touch INFO] Error connecting to Modbus device: {e}")
                print("[Touch INFO] Falling back to simulated data mode")
                self.use_simulated_data = True
                self.client = None
    
    def close(self):
        """Close the Modbus client connection if it's open."""
        if self.client is not None:
            try:
                self.client.close()
                self.client = None
                print("[Touch INFO] Closed Modbus client connection")
            except Exception as e:
                print(f"[Touch INFO] Error closing Modbus client: {e}")
    
    def read_all_data(self):
        """
        Read all touch sensor data and return it as a dictionary with lists.
        
        Returns:
            A dictionary with the following structure:
            {
                "pinky": [...],  # List of all pinky finger sensor values
                "ring": [...],   # List of all ring finger sensor values
                "middle": [...], # List of all middle finger sensor values
                "index": [...],  # List of all index finger sensor values
                "thumb": [...],  # List of all thumb sensor values
                "palm": [...]    # List of all palm sensor values
            }
        """
        # Initialize the result dictionary
        result = {finger: [] for finger in FINGER_SEGMENTS.keys()}
        
        # Read data for each finger and segment
        for finger, segments in FINGER_SEGMENTS.items():
            for segment in segments:
                segment_data = read_segment_data(
                    self.client, 
                    segment, 
                    self.use_simulated_data, 
                    self.frame
                )
                result[finger].extend(segment_data)
        
        self.frame += 1
        return result
    
    def __enter__(self):
        """Context manager entry point."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point - ensures proper cleanup."""
        self.close()


def read_all_data(client=None, use_simulated_data=False, frame=0):
    """
    Read all touch sensor data (legacy function).
    Consider using TouchSensorReader class for better connection management.
    """
    with TouchSensorReader(use_simulated_data) as reader:
        return reader.read_all_data()

class TouchDataVisualizer:
    def __init__(self, use_simulated_data=False, client=None):
        self.use_simulated_data = use_simulated_data
        self.frame_count = 0
        self.external_client = client is not None

        # Use provided client or create a new one if using real data
        if not use_simulated_data:
            if client:
                # Use the provided client
                self.client = client
                print("Using external Modbus client for touch data visualization")
                
                # Verify the client is connected
                try:
                    if not self.client.connected:
                        print("External client is not connected")
                        print("Falling back to simulated data mode")
                        self.use_simulated_data = True
                except AttributeError:
                    # Try the older API
                    try:
                        if not self.client.is_socket_open():
                            print("External client is not connected")
                            print("Falling back to simulated data mode")
                            self.use_simulated_data = True
                    except Exception as e:
                        print(f"Error checking client connection: {e}")
                        print("Falling back to simulated data mode")
                        self.use_simulated_data = True
            else:
                # Create a new client
                try:
                    self.client = ModbusTcpClient(host=MODBUS_IP, port=MODBUS_PORT)
                    # Connect the client
                    self.client.connect()
                    print(f"[Touch INFO] Successfully connected to Modbus device at {MODBUS_IP}:{MODBUS_PORT}")
                except Exception as e:
                    print(f"[Touch INFO] Error connecting to Modbus device: {e}")
                    print("[Touch INFO] Falling back to simulated data mode")
                    self.use_simulated_data = True
                    self.client = None
        else:
            print("Using simulated data mode")
            self.client = None

        # Create figure and subplots for each finger/part with hand-like layout
        self.fig = plt.figure(figsize=(18, 15))
        mode_str = "SIMULATED DATA" if self.use_simulated_data else "LIVE DATA"
        self.fig.suptitle(f"Touch Sensor Data Visualization ({mode_str})", fontsize=18)

        # Create grid for hand layout - top rows for fingers, bottom row for palm
        # Use more space between palm and fingers to prevent overlap
        self.grid = plt.GridSpec(4, 5, figure=self.fig, height_ratios=[2, 2, 1, 2])

        # Store all visualization objects
        self.segment_axes = {}  # Store axes by finger and segment name
        self.segment_images = {}  # Store image objects by finger and segment name
        self.finger_names = ["pinky", "ring", "middle", "index", "thumb", "palm"]

        # Create custom colormap (blue to red)
        colors = [
            (0, 0, 1),
            (0, 1, 1),
            (1, 1, 0),
            (1, 0, 0),
        ]  # Blue -> Cyan -> Yellow -> Red
        self.cmap = LinearSegmentedColormap.from_list("touch_cmap", colors, N=256)

        # Initialize plots based on FINGER_SEGMENTS
        self._init_finger_visualizations()

        # Add layout title for each finger
        for col, finger in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
            plt.figtext(
                0.1 + col * 0.16, 0.95, FINGER_LABELS[finger], fontsize=14, ha="center"
            )

        # Add palm title - moved lower to match the new layout
        plt.figtext(0.5, 0.25, FINGER_LABELS["palm"], fontsize=14, ha="center")

        # Initialize text objects for status information
        self.timestamp_text = self.fig.text(0.01, 0.01, "", fontsize=10)
        self.frequency_text = self.fig.text(0.35, 0.01, "", fontsize=10)
        self.status_text = self.fig.text(0.7, 0.01, "", fontsize=10)
        self.stats_text = self.fig.text(0.5, 0.97, "", fontsize=12, ha="center")

    def _init_finger_visualizations(self):
        """Create subplots for each finger segment"""
        # Set column positions for each finger
        finger_columns = {
            "thumb": 0,
            "index": 1,
            "middle": 2,
            "ring": 3,
            "pinky": 4,
            "palm": 2,  # Center column for palm
        }

        # Initialize segment plots for each finger
        for finger, segments in FINGER_SEGMENTS.items():
            col = finger_columns[finger]

            # Place palm at the bottom row, fingers at the top rows
            if finger == "palm":
                # Create one plot for palm spanning multiple columns - using the 4th row now
                ax = self.fig.add_subplot(
                    self.grid[3, 1:4]
                )  # Bottom center, span 3 columns
                segment = segments[0]  # Palm has only one segment
                dims = segment["dims"]
                empty_data = np.zeros(dims)
                img = ax.imshow(
                    empty_data,
                    cmap=self.cmap,
                    vmin=0,
                    vmax=1023,
                    interpolation="nearest",
                    aspect="auto",
                )
                ax.set_title(f"Palm", fontsize=12)

                # Store the axis and image
                self.segment_axes[(finger, "palm")] = ax
                self.segment_images[(finger, "palm")] = img

                # Add colorbar
                plt.colorbar(img, ax=ax)
            else:
                # For fingers, create a separate plot for each segment
                for i, segment in enumerate(segments):
                    # Place segments vertically: tip at top, then pad, then base
                    row = 0 if i == 0 else (1 if i == 1 else 2)
                    segment_name = segment["name"]
                    dims = segment["dims"]

                    # Create subplot
                    ax = self.fig.add_subplot(self.grid[row, col])
                    empty_data = np.zeros(dims)
                    img = ax.imshow(
                        empty_data,
                        cmap=self.cmap,
                        vmin=0,
                        vmax=1023,
                        interpolation="nearest",
                        aspect="auto",
                    )

                    # Add segment title
                    ax.set_title(f"{segment_name} ({dims[0]}x{dims[1]})")

                    # Store the axis and image
                    self.segment_axes[(finger, segment_name)] = ax
                    self.segment_images[(finger, segment_name)] = img

                    # Add colorbar
                    plt.colorbar(img, ax=ax)

        # Adjust layout
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        self.fig.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

        # Add timestamp and frequency text
        self.timestamp_text = self.fig.text(0.5, 0.01, "", ha="center")
        self.frequency_text = self.fig.text(0.98, 0.01, "", ha="right")
        self.status_text = self.fig.text(0.02, 0.01, "", ha="left")

        # Store the last data for debugging
        self.last_data = {}
        self.last_update_time = time.time()

    def update_plot(self, frame):
        start_time = time.time()
        self.frame_count = frame

        try:
            updated_artists = []
            all_values = []  # To calculate global stats

            # Process each finger and its segments
            for finger, segments in FINGER_SEGMENTS.items():
                for segment in segments:
                    segment_name = segment["name"]

                    # Read data for this segment (real or simulated)
                    segment_data = read_segment_data(
                        self.client,
                        segment,
                        use_simulated_data=self.use_simulated_data,
                        frame=frame,
                    )

                    # Keep track of all values for global stats
                    all_values.extend(segment_data)

                    # Reshape the data to match the segment dimensions
                    dims = segment["dims"]
                    reshaped_data = reshape_data(segment_data, dims)

                    # Update the image
                    img = self.segment_images[(finger, segment_name)]

                    # Transpose palm data for better visualization
                    if finger == "palm":
                        reshaped_data = reshaped_data.T

                    img.set_array(reshaped_data)
                    updated_artists.append(img)

                    # Add segment stats
                    if segment_data:
                        min_val = min(segment_data)
                        max_val = max(segment_data)
                        avg_val = sum(segment_data) / len(segment_data)

                        ax = self.segment_axes[(finger, segment_name)]
                        ax.set_xlabel(
                            f"Min: {min_val} | Max: {max_val} | Avg: {avg_val:.1f}"
                        )

            # Calculate and display global frequency and stats
            end_time = time.time()
            update_time = end_time - start_time
            frequency = 1 / update_time if update_time > 0 else 0

            # Calculate global stats
            if all_values:
                global_min = min(all_values)
                global_max = max(all_values)
                global_avg = sum(all_values) / len(all_values)
                stats_text = f"Global Min: {global_min} | Max: {global_max} | Avg: {global_avg:.1f}"
            else:
                stats_text = "No data available"

            # Create or update text objects if they don't exist
            if not hasattr(self, "timestamp_text"):
                self.timestamp_text = self.fig.text(0.01, 0.01, "", fontsize=10)
                self.frequency_text = self.fig.text(0.35, 0.01, "", fontsize=10)
                self.status_text = self.fig.text(0.7, 0.01, "", fontsize=10)
                self.stats_text = self.fig.text(0.5, 0.97, "", fontsize=12, ha="center")

            # Update information text
            self.timestamp_text.set_text(f"Time: {time.strftime('%H:%M:%S')}")
            self.frequency_text.set_text(f"Update Rate: {frequency:.2f} Hz")

            mode_text = "SIMULATED" if self.use_simulated_data else "REAL-TIME"
            self.status_text.set_text(f"Frame: {frame} ({mode_text})")
            self.stats_text.set_text(stats_text)

            # Add text objects to the list of artists to update
            updated_artists.extend(
                [
                    self.timestamp_text,
                    self.frequency_text,
                    self.status_text,
                    self.stats_text,
                ]
            )

            return updated_artists

        except Exception as e:
            print(f"Error in update_plot: {e}")
            import traceback

            traceback.print_exc()
            # Return something valid to prevent animation errors
            return list(self.segment_images.values())

    def close(self):
        try:
            # Only close the client if it's not an external client
            if self.client and not self.external_client:
                # Check if client is connected using the appropriate method
                is_connected = False
                try:
                    # Try new API first
                    is_connected = self.client.connected
                except AttributeError:
                    # Fall back to old API
                    try:
                        is_connected = self.client.is_socket_open()
                    except Exception:
                        pass
                
                if is_connected:
                    print("Closing Modbus client connection...")
                    self.client.close()
                    print("Connection closed.")
            elif self.external_client and self.client:
                print("Using external Modbus client - not closing connection")
        except Exception as e:
            print(f"Error closing client: {e}")
            import traceback

            traceback.print_exc()


def visualize_touch_data(use_simulated_data=True):
    print("Starting Touch Data Visualization")

    visualizer = TouchDataVisualizer(use_simulated_data=use_simulated_data)

    try:
        # Create animation - use blit=False to avoid the AttributeError
        ani = animation.FuncAnimation(
            visualizer.fig,
            visualizer.update_plot,
            frames=range(1000),  # Limited number of frames
            interval=100,  # Update every 100ms
            blit=False,  # Disable blitting to avoid the error
            cache_frame_data=False,  # Don't cache frame data
        )

        # Keep a reference to avoid garbage collection
        visualizer.ani = ani

        print("Animation started, showing plot...")
        plt.show()

    except Exception as e:
        print(f"Error in animation: {e}")
    finally:
        if hasattr(visualizer, "client") and visualizer.client:
            visualizer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Touch sensor data visualization")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use simulated data instead of real sensor data",
    )
    args = parser.parse_args()

    visualize_touch_data(use_simulated_data=args.simulate)
