from reedsolo import RSCodec
from public_function.public_fun import diffbits_with_two_bytes
from audio_hiding import embed_bytes, extract_bytes, count_max_window_size_by_bytestream, embed_extract_bytes, np_type
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import wave
import numpy as np


# reed编码与解码函数
def reed_encode(bytes_stream: bytes, fec_size: int):
    rs = RSCodec(fec_size)
    return rs.encode(bytes_stream)


def reed_decode(bytes_stream: bytes, fec_size: int):
    rs = RSCodec(fec_size)
    return rs.decode(bytes_stream)


def audio_info(audio_path):
    with wave.open(audio_path, 'rb') as f:
        sw = f.getsampwidth()
        fr = f.getframerate()
        fn = f.getnframes()
        nc = f.getnchannels()
        # 计算音频持续时间
        audio_time = fn / fr
        # 打印输出信息
        print("采样宽度：{} bytes".format(sw))
        print("采样频率：{} Hz".format(fr))
        print("帧数：{}".format(fn))
        print("声道数：{}".format(nc))
        print("音频时长：{:.2f} s".format(audio_time))


# 一次实验的流程
def one_exper(data_length: int, audio, output_audio=None, fec_size=20):
    if not output_audio:
        output_audio = 'output/exper_one_exper.wav'
    key = '235401 梁敏智'.encode('utf-8')
    seu_bytes = np.random.randint(0, 255, data_length).astype(np.uint8).tobytes()
    # 填充
    padding_bytes = pad(seu_bytes, AES.block_size)
    # 加密
    cipher = AES.new(key, AES.MODE_ECB)
    encryption_bytes = cipher.encrypt(padding_bytes)
    encode_bytes = reed_encode(encryption_bytes, fec_size)
    encode_bytes = bytes(encode_bytes)
    max_window_size = count_max_window_size_by_bytestream(audio, len(encode_bytes))
    # 隐藏
    data_length = embed_bytes(audio, encode_bytes, output_audio, max_window_size)
    # 提取
    ext_bytes = extract_bytes(output_audio, data_length, max_window_size)
    try:
        # 解码
        decode_bytes = reed_decode(ext_bytes, fec_size)[0]
        decode_bytes = bytes(decode_bytes)
        # 解密
        decryption_bytes = cipher.decrypt(decode_bytes)
        # 去填充
        decryption_bytes = unpad(decryption_bytes, AES.block_size)
        diff_2, length_2 = diffbits_with_two_bytes(seu_bytes, decryption_bytes)
        diff, length = diffbits_with_two_bytes(encode_bytes, ext_bytes)
        return diff, length, max_window_size, diff_2, length_2
    except Exception as e:
        # print(e)
        result = '长度为{}时发生错误'.format(data_length)
        print(result)
        diff_2 = 0
        length_2 = data_length
        diff, length = diffbits_with_two_bytes(encode_bytes, ext_bytes)
        return diff, length, max_window_size, diff_2, length_2


# 分析误码率
# 101, 3400, 3900, 5700, 4700, 4900
def bits_error_rate_analysis(audio_path):
    # length 为100字节到3000字节，步长为100字节
    # length = [100, 101, 102, 110, 4000]
    # length = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
    length1 = [100, 101, 102, 110, 3399, 3400, 3401, 3402]
    length2 = [i for i in range(100, 5000, 200)]
    # 拼接两个length
    length = length1 + length2
    # length = [3626] # 发生严重错误
    print("误码率：\n数据长度/bits\t\t嵌入提取误码率/比特级\t\t总误码率/比特级别\t\t窗口大小")
    for l in length:
        diff, le, w, diff2, le2 = one_exper(l, audio_path)
        print("{}\t\t\t\t{}/{}={:.2f}%\t\t\t\t{}/{}={:.2f}%\t\t\t\t{}".format(l*8, diff, le, diff / le * 100, diff2, le2,
                                                                              diff2 / le2 * 100, w))


def bits_error_rate_analysis2(audio_path):
    length = [i for i in range(100, 9000, 200)]
    # length = [95, 106]
    print("误码率：\n数据长度/bits\t\t误码率/比特级\t\t窗口大小")
    for l in length:
        w, r = embed_extract_bytes(audio_path, l, False)
        print("{}\t\t{:.2f}\t\t{}".format(l * 8, 100 * r, w))


def calculate_snr(origin_audio_path, output_audio_path):
    """
    计算信噪比
    :param audio_path: 音频文件，默认就是用 启动_big.wav
    :param image_path: 图片文件，默认就是用 seu_light.png
    :param mode: 嵌入模式，默认就是用 adaptive_LSB_nosort
    :return: 信噪比
    """

    # 读取原始音频文件
    original_audio = wave.open(origin_audio_path, 'rb')
    original_audio_frames = original_audio.readframes(original_audio.getnframes())
    original_audio_array = np.frombuffer(original_audio_frames, dtype=np_type)

    # 读取嵌入信息后的音频文件
    embedded_audio = wave.open(output_audio_path, 'rb')
    embedded_audio_frames = embedded_audio.readframes(embedded_audio.getnframes())
    embedded_audio_array = np.frombuffer(embedded_audio_frames, dtype=np_type)

    # 计算信噪比
    signal_power = np.sum(original_audio_array ** 2)
    noise_power = np.sum((original_audio_array - embedded_audio_array) ** 2)

    snr = 10 * np.log10(signal_power / noise_power)

    return snr

def calculate_mse_psnr(origin_audio_path, output_audio_path):
    # 计算mse
    original_audio = wave.open(origin_audio_path, 'rb')
    original_audio_frames = original_audio.readframes(original_audio.getnframes())
    original_audio_array = np.frombuffer(original_audio_frames, dtype=np_type)

    # 读取嵌入信息后的音频文件
    embedded_audio = wave.open(output_audio_path, 'rb')
    embedded_audio_frames = embedded_audio.readframes(embedded_audio.getnframes())
    embedded_audio_array = np.frombuffer(embedded_audio_frames, dtype=np_type)

    # 计算均方误差 (MSE)
    mse = np.mean((original_audio_array - embedded_audio_array) ** 2)

    # 计算峰值信噪比 (PSNR)
    max_possible_power = np.max(original_audio_array) - np.min(original_audio_array)
    psnr = 10 * np.log10((max_possible_power ** 2) / mse)

    return mse, psnr

# 填充 -> 加密 -> 编码 -> 隐藏 -> 提取 -> 解码 -> 解密 -> 去填充
if __name__ == '__main__':
    audio = '启动.wav'
    output_audio = 'output/string_exper.wav'
    seu_str = '2023年网络信息安全与信息隐藏课程，东南大学网络空间安全学院'
    key = '235401 梁敏智'.encode('utf-8')
    seu_bytes = seu_str.encode('utf-8')

    audio_info(audio)

    # 填充
    padding_bytes = pad(seu_bytes, AES.block_size)

    # 加密
    cipher = AES.new(key, AES.MODE_ECB)
    encryption_bytes = cipher.encrypt(padding_bytes)
    fec_size = 20

    encode_bytes = reed_encode(encryption_bytes, fec_size)
    encode_bytes = bytes(encode_bytes)

    max_window_size = count_max_window_size_by_bytestream(audio, len(encode_bytes))
    print('max_window_size: ', max_window_size)
    # mode = 'adaptive_LSB_nosort'

    # 隐藏
    data_length = embed_bytes(audio, encode_bytes, output_audio, max_window_size)

    # 提取
    ext_bytes = extract_bytes(output_audio, data_length, max_window_size)

    # 比较差异
    diff, length = diffbits_with_two_bytes(encode_bytes, ext_bytes)

    # 解码
    decode_bytes = reed_decode(ext_bytes, fec_size)[0]
    decode_bytes = bytes(decode_bytes)

    # 解密
    decryption_bytes = cipher.decrypt(decode_bytes)

    # 去填充
    decryption_bytes = unpad(decryption_bytes, AES.block_size)

    # 格式化打印所有过程的字节流长度以及字节流
    print('名称\t长度/bits\t值')
    print('原始\t{}\t{}'.format(len(seu_bytes) * 8, seu_bytes))
    print('填充\t{}\t{}'.format(len(padding_bytes) * 8, padding_bytes))
    print('加密\t{}\t{}'.format(len(encryption_bytes) * 8, encryption_bytes))
    print('编码\t{}\t{}'.format(len(encode_bytes) * 8, encode_bytes))
    print('提取\t{}\t{}'.format(len(ext_bytes) * 8, ext_bytes))
    print('解码\t{}\t{}'.format(len(decode_bytes) * 8, decode_bytes))
    print('解密\t{}\t{}'.format(len(decryption_bytes) * 8, decryption_bytes))
    print('================================================')

    srcdiff, srclength = diffbits_with_two_bytes(seu_bytes, decryption_bytes)
    print('原始字节流与解密字节流的差异：{}/{}={}%'.format(srcdiff, srclength, srcdiff / srclength * 100))
    encodediff, encodelength = diffbits_with_two_bytes(encode_bytes, ext_bytes)
    print('编码字节流与提取字节流的差异：{}/{}={}%'.format(encodediff, encodelength, encodediff / encodelength * 100))

    print('最终提取结果: ', decryption_bytes.decode('utf-8'))

    # 打印信噪比
    snr = calculate_snr(audio, output_audio)
    print('信噪比为：', snr)

    # 打印均方误差和峰值信噪比
    mse, psnr = calculate_mse_psnr(audio, output_audio)
    print('均方误差为：', mse)
    print('峰值信噪比为：', psnr)

    print('================================================')
    # bits_error_rate_analysis(audio)
    print('================================================')
    # bits_error_rate_analysis2(audio)

    print('================================================')
    # 高斯噪声测试
    with wave.open(output_audio, 'rb') as f:
        audio_frames = f.readframes(f.getnframes())
        audio_array = np.frombuffer(audio_frames, dtype=np_type).copy()
        # noise_level = [i*0.000001 for i in range(1, 10)]
        noise_level = [1e-5 * 0.8, 0.9 * 1e-5, 1e-5, 1e-5*2]
        for n in noise_level:
            noise_audio_array = audio_array.copy()
            noise = np.random.normal(-n*32767, n*23767, len(audio_array)).astype(np_type)
            noise_audio_array += noise
            noise_audio_frames = noise_audio_array.tobytes()
            with wave.open('output/noise.wav', 'wb') as f2:
                f2.setparams(f.getparams())
                f2.writeframes(noise_audio_frames)

            # 提取，解密，解码，去填充，计算误码率
            ext_bytes = extract_bytes('output/noise.wav', data_length, max_window_size)

            try:
                decode_bytes = reed_decode(ext_bytes, fec_size)[0]
                decode_bytes = bytes(decode_bytes)
                decryption_bytes = cipher.decrypt(decode_bytes)
                decryption_bytes = unpad(decryption_bytes, AES.block_size)
                srcdiff, srclength = diffbits_with_two_bytes(seu_bytes, decryption_bytes)
            except Exception as e:
                srcdiff = 100
                srclength = len(seu_bytes)
            diff, length = diffbits_with_two_bytes(ext_bytes, encode_bytes)

            # 打印结果
            print('噪声水平：', n)
            print('原始字节流与解密字节流的差异：{}/{}={}%'.format(srcdiff, srclength, srcdiff / srclength * 100))
            print('编码字节流与提取字节流的差异：{}/{}={}%'.format(diff, length, diff / length * 100))
    print('==========白噪声===============')
    # 白噪声测试
    with wave.open(output_audio, 'rb') as f:
        audio_frames = f.readframes(f.getnframes())
        audio_array = np.frombuffer(audio_frames, dtype=np_type).copy()
        # noise_level = [i*0.00001 for i in range(1, 10)]
        noise_level = [3*1e-5,3.05*1e-5,3.1*1e-5, 3.15*1e-5]
        for n in noise_level:
            noise_audio_array = audio_array.copy()
            noise = np.random.uniform(-n*32767, n*23767, len(audio_array)).astype(np_type)
            noise_audio_array += noise
            noise_audio_frames = noise_audio_array.tobytes()
            with wave.open('output/noise.wav', 'wb') as f2:
                f2.setparams(f.getparams())
                f2.writeframes(noise_audio_frames)

            # 提取，解密，解码，去填充，计算误码率
            ext_bytes = extract_bytes('output/noise.wav', data_length, max_window_size)

            try:
                decode_bytes = reed_decode(ext_bytes, fec_size)[0]
                decode_bytes = bytes(decode_bytes)
                decryption_bytes = cipher.decrypt(decode_bytes)
                decryption_bytes = unpad(decryption_bytes, AES.block_size)
                srcdiff, srclength = diffbits_with_two_bytes(seu_bytes, decryption_bytes)
            except Exception as e:
                srcdiff = 100
                srclength = len(seu_bytes)
            diff, length = diffbits_with_two_bytes(ext_bytes, encode_bytes)

            # 打印结果
            print('噪声水平：', n)
            print('原始字节流与解密字节流的差异：{}/{}={}%'.format(srcdiff, srclength, srcdiff / srclength * 100))
            print('编码字节流与提取字节流的差异：{}/{}={}%'.format(diff, length, diff / length * 100))



