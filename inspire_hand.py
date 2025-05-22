import serial
import struct
import time
#import numpy 
#import string
#import binascii

global hand_id
hand_id = 0xff

#把数据分成高字节和低字节
def data2bytes(data):
    rdata = [0xff]*2
    if data == -1:
        rdata[0] = 0xff
        rdata[1] = 0xff
    else:
        rdata[0] = data&0xff
        rdata[1] = (data>>8)&(0xff)
    return rdata

#把十六进制或十进制的数转成bytes
def num2str(num):
    str = hex(num)
    str = str[2:4]
    if(len(str) == 1):
        str = '0'+ str
    str = bytes.fromhex(str)     
    #print(str)
    return str

#求校验和
def checknum(data,leng):
    result = 0
    for i in range(2,leng):
        result += data[i]
    result = result&0xff
    #print(result)
    return result

#设置角度
def setangle(angle1,angle2,angle3,angle4,angle5,angle6):
    global hand_id
    if angle1 <-1 or angle1 >1000:
        print('数据超出正确范围：-1-1000')
        return
    if angle2 <-1 or angle2 >1000:
        print('数据超出正确范围：-1-1000')
        return
    if angle3 <-1 or angle3 >1000:
        print('数据超出正确范围：-1-1000')
        return
    if angle4 <-1 or angle4 >1000:
        print('数据超出正确范围：-1-1000')
        return
    if angle5 <-1 or angle5 >1000:
        print('数据超出正确范围：-1-1000')
        return
    if angle6 <-1 or angle6 >1000:
        print('数据超出正确范围：-1-1000')
        return
    
    datanum = 0x0F
    b = [0]*(datanum + 5)
    #包头
    b[0] = 0xEB
    b[1] = 0x90

    #hand_id号
    b[2] = hand_id

    #数据个数
    b[3] = datanum
    
    #写操作
    b[4] = 0x12
    
    #地址
    b[5] = 0xCE
    b[6] = 0x05
     
    #数据
    b[7] = data2bytes(angle1)[0]
    b[8] = data2bytes(angle1)[1]
    
    b[9] = data2bytes(angle2)[0]
    b[10] = data2bytes(angle2)[1]
    
    b[11] = data2bytes(angle3)[0]
    b[12] = data2bytes(angle3)[1]
    
    b[13] = data2bytes(angle4)[0]
    b[14] = data2bytes(angle4)[1]
    
    b[15] = data2bytes(angle5)[0]
    b[16] = data2bytes(angle5)[1]
    
    b[17] = data2bytes(angle6)[0]
    b[18] = data2bytes(angle6)[1]
    
    #校验和
    b[19] = checknum(b,datanum+4)
    
    #向串口发送数据
    putdata = b''
    
    for i in range(1,datanum+6):
        putdata = putdata + num2str(b[i-1])
    ser.write(putdata)
    print('发送的数据：')
    for i in range(1,datanum+6):
        print(hex(putdata[i-1]))
    
    getdata= ser.read(9)
    print('返回的数据：')
    for i in range(1,10):
        print(hex(getdata[i-1]))



#串口设置
ser=serial.Serial('/dev/ttyUSB0',115200)
ser.isOpen()

# 主程序
if __name__ == "__main__":
    # 定义要设置的角度值
    angles = [
        (1000, 1000, 1000, 1000, 1000, 1000), 
        (500, 500, 500, 500, 500, 500),      
        (0, 0, 0, 0, 0, 1000)   
    ]

    # 循环调用 setangle 函数
    for angle_set in angles:
        setangle(*angle_set)  # 解包角度值
        time.sleep(2)  # 等待 2 秒


