import time
# from pymodbus.client import ModbusTcpClient
from pymodbus.client.sync import ModbusTcpClient
from pymodbus.pdu import ExceptionResponse

# 定义 Modbus TCP 相关参数
MODBUS_IP = "192.168.11.210"
MODBUS_PORT = 6000

# 定义各部分数据地址范围
TOUCH_SENSOR_BASE_ADDR_PINKY = 3000  # 小拇指
TOUCH_SENSOR_END_ADDR_PINKY = 3369

TOUCH_SENSOR_BASE_ADDR_RING = 3370  # 无名指
TOUCH_SENSOR_END_ADDR_RING = 3739

TOUCH_SENSOR_BASE_ADDR_MIDDLE = 3740  # 中指
TOUCH_SENSOR_END_ADDR_MIDDLE = 4109

TOUCH_SENSOR_BASE_ADDR_INDEX = 4110  # 食指
TOUCH_SENSOR_END_ADDR_INDEX = 4479

TOUCH_SENSOR_BASE_ADDR_THUMB = 4480  # 大拇指
TOUCH_SENSOR_END_ADDR_THUMB = 4899

TOUCH_SENSOR_BASE_ADDR_PALM = 4900  # 掌心
TOUCH_SENSOR_END_ADDR_PALM = 5123

# Modbus 每次最多读取寄存器的数量
MAX_REGISTERS_PER_READ = 125


def read_register_range(client, start_addr, end_addr):
    """
    批量读取指定地址范围内的寄存器数据。
    """
    register_values = []  
    # 分段读取寄存器
    for addr in range(start_addr, end_addr + 1, MAX_REGISTERS_PER_READ * 2):

        current_count = min(MAX_REGISTERS_PER_READ, (end_addr - addr) // 2 + 1)


        response = client.read_holding_registers(address=addr, count=current_count)

        if isinstance(response, ExceptionResponse) or response.isError():
            print(f"读取寄存器 {addr} 失败: {response}")
            register_values.extend([0] * current_count)  
        else:
            register_values.extend(response.registers) 

    return register_values


def read_multiple_registers():
    client = ModbusTcpClient(MODBUS_IP, port=MODBUS_PORT)
    client.connect()

    try:
        while True:  
            start_time = time.time()  

            # 读取各部分数据
            pinky_register_values = read_register_range(
                client,
                TOUCH_SENSOR_BASE_ADDR_PINKY,
                TOUCH_SENSOR_END_ADDR_PINKY
            )

            ring_register_values = read_register_range(
                client,
                TOUCH_SENSOR_BASE_ADDR_RING,
                TOUCH_SENSOR_END_ADDR_RING
            )

            middle_register_values = read_register_range(
                client,
                TOUCH_SENSOR_BASE_ADDR_MIDDLE,
                TOUCH_SENSOR_END_ADDR_MIDDLE
            )

            index_register_values = read_register_range(
                client,
                TOUCH_SENSOR_BASE_ADDR_INDEX,
                TOUCH_SENSOR_END_ADDR_INDEX
            )

            thumb_register_values = read_register_range(
                client,
                TOUCH_SENSOR_BASE_ADDR_THUMB,
                TOUCH_SENSOR_END_ADDR_THUMB
            )

            palm_register_values = read_register_range(
                client,
                TOUCH_SENSOR_BASE_ADDR_PALM,
                TOUCH_SENSOR_END_ADDR_PALM
            )

            end_time = time.time()
            frequency = 1 / (end_time - start_time) 

            pinky_output_str = ", ".join(map(str, pinky_register_values))
            ring_output_str = ", ".join(map(str, ring_register_values))
            middle_output_str = ", ".join(map(str, middle_register_values))
            index_output_str = ", ".join(map(str, index_register_values))
            thumb_output_str = ", ".join(map(str, thumb_register_values))
            palm_output_str = ", ".join(map(str, palm_register_values))

            # 打印数据
            print(f"小拇指数据：{pinky_output_str}")
            print(f"无名指数据：{ring_output_str}")
            print(f"中指数据：{middle_output_str}")
            print(f"食指数据：{index_output_str}")
            print(f"大拇指数据：{thumb_output_str}")
            print(f"掌心数据：{palm_output_str}")
            print(f"读取频率：{frequency:.2f} Hz")  

    finally:
        client.close()  


if __name__ == "__main__":
    read_multiple_registers()

