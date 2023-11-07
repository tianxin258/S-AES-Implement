def subNibbles(input):
    # S-Box
    sBox = [0x9, 0x4, 0xA, 0xB, 0xD, 0x1, 0x8, 0x5,
            0x6, 0x2, 0x0, 0x3, 0xC, 0xE, 0xF, 0x7]
    # 返回替代后的值
    return sBox[input]
def ShiftRow(input):
    # 分离前八位和后八位
    upper_byte = input >> 8
    lower_byte = input & 0xFF
    
    # 前四位和后四位互相交换
    upper_nibble = lower_byte >> 4
    lower_nibble = lower_byte & 0x0F
    swapped_lower_byte = (lower_nibble << 4) | upper_nibble
    
    # 合并前八位和交换后的后八位
    result = (upper_byte << 8) | swapped_lower_byte

    return result

def mixColumns(state):
    # 定义列混淆矩阵
    mix_matrix = [[1, 4],
                  [4, 1]]
    # 定义有限域乘法
    def mul(a, b):
        p = 0
        for c in range(4):  # 在GF(2^4)中，我们只需要迭代4次
            if b & 1:
                p ^= a
            hi_bit_set = a & 0x8  # 在GF(2^4)中，最高位是第4位
            a <<= 1
            if hi_bit_set:
                a ^= 0x13  # 在GF(2^4)中，如果最高位为1，那么左移后需要异或上0x13
            b >>= 1
        return p % 16  # 在GF(2^4)中，结果需要模16
    # 对状态矩阵的每一列进行列混淆
    a = [[0]*2 for _ in range(2)]
    for i in range(2):  # AES state is 2x2 in S-AES
        for j in range(2):
           a[i][j] = mul(mix_matrix[i][0], state[0][j]) ^ mul(mix_matrix[i][1], state[1][j])
    return a
def keyGeneration(key):
    w = [None]*6
    w[0] = (key & 0xFF00) >> 8
    w[1] = key & 0x00FF
    # Substitution for w2
    w[2] = w[0] ^ (subNibbles((w[1] & 0xF0) >> 4) << 4 | subNibbles(w[1] & 0x0F)) ^ 0x80
    w[3] = w[2] ^ w[1]
    # Substitution for w4
    w[4] = w[2] ^ (subNibbles((w[3] & 0xF0) >> 4) << 4 | subNibbles(w[3] & 0x0F)) ^ 0x30
    w[5] = w[4] ^ w[3]
    return w
def invSubNibbles(input):
    # Inverse S-Box
    inv_sBox = [0xA, 0x5, 0x9, 0xB, 0x1, 0x7, 0x8, 0xF,
                0x6, 0x0, 0x2, 0x3, 0xC, 0x4, 0xD, 0xE]
    # 返回替代后的值
    return inv_sBox[input]
def inverseShiftRow(input):
    # 分离前八位和后八位
    upper_byte = input >> 8
    lower_byte = input & 0xFF
    # 前四位和后四位互相交换
    upper_nibble = lower_byte >> 4
    lower_nibble = lower_byte & 0x0F
    swapped_lower_byte = (lower_nibble << 4) | upper_nibble
    # 合并前八位和交换后的后八位
    result = (upper_byte << 8) | swapped_lower_byte
    return result

def inverseMixColumns(state):
    mix_matrix = [[9, 2],
                      [2, 9]]
    
    def mul(a, b):
        p = 0
        for _ in range(4):
            if b & 1:
                p ^= a
            hi_bit_set = a & 0x8
            a <<= 1
            if hi_bit_set:
                a ^= 0x13
            b >>= 1
        return p % 16
    
    a = [[0]*2 for _ in range(2)]
    for i in range(2):  # AES state is 2x2 in S-AES
        for j in range(2):
           a[i][j] = mul(mix_matrix[i][0], state[0][j]) ^ mul(mix_matrix[i][1], state[1][j])
    return a
def int_to_hex_matrix(input):
    # 将输入转化为4位的十六进制数
    hex_input = "{:04x}".format(input)
    # 创建一个2x2的矩阵
    matrix = [[0]*2 for _ in range(2)]
    # 将十六进制数的每一位放入矩阵中
    for i in range(4):
        matrix[i//2][i%2] = int(hex_input[i],16)
    return matrix
def hex_matrix_to_int(matrix):
    hex_str=0
    # 将矩阵中的每一位拼接成一个字符串
    for i in range(4):
        hex_str += matrix[i//2][i%2]*pow(16,3-i)
    # 将字符串转化为整数
    return hex_str

def sAES(input, key):
    w = keyGeneration(key)
    # print(w)
    # Add round key 1 (w0, w1)
    input ^= (w[0] << 8 | w[1])
    # Substitution
    input = subNibbles((input & 0xF000) >> 12) << 12 | subNibbles((input & 0xF00) >> 8) << 8 | subNibbles((input & 0xF0) >> 4) << 4 | subNibbles(input & 0x0F)
    # print(input)
    # Shift row
    input = ShiftRow(input)
    # print(input)
    state=int_to_hex_matrix(input)
    state = mixColumns(state)
    # print(state)
    state=hex_matrix_to_int(state)
    # Add round key 2 (w2, w3)
    input ^= (w[2] << 8 | w[3])
    # print(input)
    # Substitution
    input = subNibbles((input & 0xF000) >> 12) << 12 | subNibbles((input & 0xF00) >> 8) << 8 | subNibbles((input & 0xF0) >> 4) << 4 | subNibbles(input & 0x0F)
    # print(input)
    input = ShiftRow(input)
    # print(input)
    # Add round key 3 (w4, w5)
    input ^= (w[4] << 8 | w[5])
    return input
def invSAES(input, key):
    w = keyGeneration(key)
    # Add round key 3 (w4, w5)
    input ^= (w[4] << 8 | w[5])
    # print(input)
    # Inverse Shift row
    input = inverseShiftRow(input)
    # Inverse Substitution
    input = invSubNibbles((input & 0xF000) >> 12) << 12 | invSubNibbles((input & 0xF00) >> 8) << 8 | invSubNibbles((input & 0xF0) >> 4) << 4 | invSubNibbles(input & 0x0F)
    # print(input)
    # Add round key 2 (w2, w3)
    input ^= (w[2] << 8 | w[3])
    # Inverse Mix columns
    state=int_to_hex_matrix(input)
    state = inverseMixColumns(state)
    # print(state)
    state=hex_matrix_to_int(state)
    # Convert state matrix back to integer
    # Inverse Shift row
    input = inverseShiftRow(input)
    # print(input)
    # Inverse Substitution
    input = invSubNibbles((input & 0xF000) >> 12) << 12 | invSubNibbles((input & 0xF00) >> 8) << 8 | invSubNibbles((input & 0xF0) >> 4) << 4 | invSubNibbles(input & 0x0F)
    # Add round key 1 (w0, w1)
    input ^= (w[0] << 8 | w[1])
    return input


def string_to_blocks(input_string):
    # 将输入字符串转换为字节
    input_bytes = input_string.encode('utf-8')

    # 将字节分割为2字节的块，并转换为16位的整数列表
    blocks = []
    for i in range(0, len(input_bytes), 2):
        # 使用0填充，确保每个块都有2字节
        block = input_bytes[i:i+2].ljust(2, b'\x00')
        blocks.append(int.from_bytes(block, 'big'))
    return blocks

def blocks_to_string(blocks):
    # 将16位整数列表转换回字符串
    output_bytes = bytearray()
    for block in blocks:
        output_bytes.extend(block.to_bytes(2, 'big'))
    # 删除填充的0并转换为字符串
    return output_bytes.rstrip(b'\x00').decode('utf-8', errors='replace')

# 修改后的sAES函数，可以接受16位整数的列表
def sAES_blocks(blocks, key):
    encrypted_blocks = []
    for block in blocks:
        encrypted_blocks.append(sAES(block, key))
    return encrypted_blocks

# 修改后的invSAES函数，可以接受16位整数的列表
def invSAES_blocks(blocks, key):
    decrypted_blocks = []
    for block in blocks:
        decrypted_blocks.append(invSAES(block, key))
    return decrypted_blocks

# 定义明文和密钥
plaintext_string = "信息安全导论"
key = 0x12bf # 密钥

# 将明文字符串转换为16位整数的列表
plaintext_blocks = string_to_blocks(plaintext_string)

# 使用S-AES进行加密
ciphertext_blocks = sAES_blocks(plaintext_blocks, key)

# 打印密文块的十六进制表示
print("The ciphertext blocks are: ")
for block in ciphertext_blocks:
    print(hex(block))

# 将加密后的块转换回字符串表示（注意，这通常是不可读的）
ciphertext_string = blocks_to_string(ciphertext_blocks)
print("The ciphertext (likely unreadable): ", ciphertext_string)

# 解密
decrypted_blocks = invSAES_blocks(ciphertext_blocks, key)

# 打印解密后的块的十六进制表示
print("The decrypted blocks are: ")
for block in decrypted_blocks:
    print(hex(block))

# 将解密后的块转换回字符串
nciphertext_string = blocks_to_string(decrypted_blocks)
print("The decrypted text is: ", nciphertext_string)

# 定义明文和密钥
plaintext = 0x254a  
key1 = 0x12bf 
key2= 0x34ae
# 使用S-AES进行加密
middletext = sAES(plaintext, key1)
ciphertext= sAES(middletext, key2)
# 打印密文

print("The middletext is: ", hex(middletext))
print("The ciphertext is: ", hex(ciphertext))

import threading
import time


KEY_SPACE = range(0, 65536)

# 存储加密的中间结果
encryption_dict = {}
encryption_dict_lock = threading.Lock()  # 创建锁以同步对字典的访问
matched_key=[]

# 加密的工作线程
def encrypt_work(start_key, end_key, plaintext):
    for key1 in range(start_key, end_key):
        mid_ciphertext = sAES(plaintext, key1)
        with encryption_dict_lock:
            encryption_dict[mid_ciphertext] = key1

# 解密的工作线程
def decrypt_work(start_key, end_key, ciphertext):
    for key2 in range(start_key, end_key):
        mid_plaintext = invSAES(ciphertext, key2)
        with encryption_dict_lock:
            if mid_plaintext in encryption_dict:
                key1 = encryption_dict[mid_plaintext]
                print(f"Matching keys found: K1 = {hex(key1)}, K2 = {hex(key2)}")
                matched_key.append((key1,key2))

                

# 创建线程的函数
def create_threads(thread_count, work_function, start_key, end_key, text):
    threads = []
    keys_per_thread = (end_key - start_key) // thread_count
    for i in range(thread_count):
        # 计算每个线程的起始和结束密钥
        thread_start = start_key + i * keys_per_thread
        thread_end = thread_start + keys_per_thread if i < thread_count - 1 else end_key
        thread = threading.Thread(target=work_function, args=(thread_start, thread_end, text))
        threads.append(thread)
        thread.start()
    return threads

# 使用多线程进行中间相遇攻击
def meet_in_the_middle_attack(plaintext, ciphertext, thread_count):
    # 分为加密和解密两部分线程
    encrypt_threads = create_threads(thread_count, encrypt_work, 0, 65536, plaintext)
    decrypt_threads = create_threads(thread_count, decrypt_work, 0, 65536, ciphertext)
    return encrypt_threads + decrypt_threads  # 返回所有线程

# 开始时间
start_time = time.time()

# 启动线程
threads = meet_in_the_middle_attack(plaintext, ciphertext, 4)  # 假设我们使用4个线程

# 等待所有线程完成
for thread in threads:
    thread.join()

# 结束时间
end_time = time.time()
print(f"Time taken for meet-in-the-middle attack: {end_time - start_time} seconds")
