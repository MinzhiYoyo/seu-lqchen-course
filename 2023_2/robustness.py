from PIL import Image
from public_function.public_fun import png_to_gray_image
from audio_hiding import embed_extract_bytes, embed_image, extract_image, diff_with_two_image, np_type, count_max_window_size
import wave
import numpy as np
from pydub import AudioSegment
import random

output = 'output/'


# 分析误码率
def bits_error_rate_analysis(audio_path):
    # length = [i*1024 for i in range(1, 21, 2)]
    length = [95, 106]
    print("误码率：\n数据长度/bits\t\t误码率/比特级\t\t窗口大小")
    for l in length:
        w, r = embed_extract_bytes(audio_path, l, False)
        print("{}\t\t{:.2f}\t\t{}".format(l*8, 100 * r, w))

    # data_size = []
    # error_rate = []
    # window_size = []
    # output_str = "数据长度/字节\t\t误码率/比特级\t\t窗口大小\n"
    # for d in tqdm.tqdm(range(1000, 102400, 500)):
    #     winsize, rate = embed_extract_bytes('启动_big.wav', d, False)
    #     data_size.append(d)
    #     error_rate.append(rate)
    #     window_size.append(winsize)
    #     output_str += str(d) + '\t\t' + str(rate) + '\t\t' + str(winsize) + '\n'
    # with open('误码率结果.txt', 'w') as f:
    #     f.write(output_str)


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


# 计算均方误差与峰值信噪比
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


# 滤波攻击测试
def filter_attack_test(image_path, output_audio_path, image_size, window_size, filter_type, cutoff_frequency=2000,
                       width_height=(25, 25), mode='adaptive_LSB_nosort', hight_frequency=20000):
    audio = AudioSegment.from_wav(output_audio_path)

    if filter_type == 'low_pass':
        filter_audio = audio.low_pass_filter(cutoff_frequency)
    elif filter_type == 'high_pass':
        filter_audio = audio.high_pass_filter(cutoff_frequency)
    elif filter_type == 'band_pass':
        filter_audio = audio.high_pass_filter(hight_frequency).low_pass_filter(cutoff_frequency)
    else:
        raise ValueError("滤波类型不正确")

    filter_audio.export(output + 'filter_{}.wav'.format(filter_type), format='wav')

    # 再次提取
    extract_image(output + 'filter_{}.wav'.format(filter_type), output + 'filter_{}.jpeg'.format(filter_type),
                  image_size, window_size, mode, width_height=width_height)

    # 比较图片的差异返回
    return diff_with_two_image(image_path, output + 'filter_{}.jpeg'.format(filter_type))


# 白噪声测试
def add_noise(image_path, output_audio_path, image_size, window_size, width_height=(25, 25), mode='adaptive_LSB_nosort',
              noise_level=0.005, noise_length=0.5, noise_mode='white_noise'):
    with wave.open(output_audio_path, 'rb') as audio_file:
        audio = np.frombuffer(audio_file.readframes(audio_file.getnframes()), dtype=np_type).copy()
        params = audio_file.getparams()
    max_audio = float(np.max(audio))
    if noise_mode == 'white_noise':
        noise = np.random.uniform(-noise_level * max_audio, noise_level * max_audio,
                                  int(len(audio) * noise_length)).astype(np_type)
    elif noise_mode == 'gaussian_noise':
        noise = np.random.normal(-noise_level * max_audio, noise_level * max_audio,
                                 int(len(audio) * noise_length)).astype(np_type)
    else:
        raise ValueError('噪声类型不正确')

    # 随机选取一段音频赋值
    random_index = random.randint(0, len(audio) - len(noise))
    audio[random_index:random_index + len(noise)] += noise

    # noise = np.random.choice(noise, len(audio), replace=False)

    # 保存
    with wave.open(output + 'noise_{}_{}_{}.wav'.format(noise_level, noise_length, noise_mode), 'wb') as audio_file:
        audio_file.setparams(params)
        audio_file.writeframes(audio.tobytes())

    # 再次提取
    extract_image(output + 'noise_{}_{}_{}.wav'.format(noise_level, noise_length, noise_mode),
                  output + 'noise_{}_{}_{}.jpeg'.format(noise_level, noise_length, noise_mode), image_size, window_size, mode,
                  width_height=width_height)

    # 比较图片的差异返回
    return diff_with_two_image(image_path, output + 'noise_{}_{}_{}.jpeg'.format(noise_level, noise_length, noise_mode))

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

def image_info(image_path):
    image = Image.open(image_path)
    w_h = image.size
    print("图片宽高：{} * {}".format(w_h[0], w_h[1]))
    # 图片位深度
    print("图片位深度：{}".format(image.bits))


if __name__ == "__main__":
    audio_path = '启动_mid.wav'
    src_image = 'seu_80.jpg'
    image_path = output+'seu_gray.jpeg'
    # 先将图片灰度化
    png_to_gray_image(src_image, image_path)
    # 模式3，自适应lsb，且不需要传递嵌入位置
    mode3 = 'adaptive_LSB_nosort'
    mode3_audio_path = output+'embed_audio_output.wav'  # 带有隐藏信息的音频路径
    mode3_extract_path = output+'extract_image.jpeg'  # 提取出来的图片路径
    mode3_window_size = count_max_window_size(audio_path, image_path, False if mode3 == 'adaptive_LSB_nosort' else True)
    print("能量窗口设置为：{}".format(mode3_window_size))

    # 获取输入图的宽高
    image = Image.open(image_path)
    w_h = image.size

    # 音频信息
    audio_info(audio_path)
    # 图片信息
    image_info(image_path)

    # 模式3，实验
    image_size = embed_image(audio_path, image_path, mode3_audio_path, mode3_window_size, embedded_mode=mode3)
    extract_image(mode3_audio_path, mode3_extract_path, image_size, mode3_window_size, mode3, width_height=w_h)
    # 比较两张图的差异性
    print("源图像与目标图像的差异：{:.2f} %".format(diff_with_two_image(image_path, mode3_extract_path) * 100))

    ################  健壮性分析   #################################

    # # 计算信噪比
    # snr = calculate_snr(audio_path, mode3_audio_path)
    # print("信噪比：{:.2f} dB".format(snr))
    #
    # # 计算均方误差与峰值信噪比
    # mse, psnr = calculate_mse_psnr(audio_path, mode3_audio_path)
    # print("均方误差：{:.2f} ，峰值信噪比：{:.2f} dB".format(mse, psnr))
    #
    # # 滤波攻击测试
    # low_freq = 15000
    # high_freq = 100
    # # 低通滤波
    # low_pass_diff = filter_attack_test(image_path, mode3_audio_path, image_size, mode3_window_size, 'low_pass', cutoff_frequency=low_freq, width_height=w_h, mode=mode3)
    # print("低通滤波后的图像差异：{:.2f} %".format(low_pass_diff*100))
    # # 高通滤波
    # high_pass_diff = filter_attack_test(image_path, mode3_audio_path, image_size, mode3_window_size, 'high_pass', cutoff_frequency=high_freq, width_height=w_h, mode=mode3)
    # print("高通滤波后的图像差异：{:.2f} %".format(high_pass_diff*100))
    # # 带通滤波
    # band_pass_diff = filter_attack_test(image_path, mode3_audio_path, image_size, mode3_window_size, 'band_pass', cutoff_frequency=low_freq, width_height=w_h, mode=mode3, hight_frequency=high_freq)
    # print("带通滤波后的图像差异：{:.2f} %".format(band_pass_diff*100))
    #
    # # 噪声测试
    # noise_test = [
    #     {
    #         'noise_level': 0.005,
    #         'noise_length': 0.1,
    #         'noise_type': 'white_noise'
    #     },
    #     {
    #         'noise_level': 0.05,
    #         'noise_length': 0.1,
    #         'noise_type': 'white_noise'
    #     },
    #     {
    #         'noise_level': 0.005,
    #         'noise_length': 0.3,
    #         'noise_type': 'white_noise'
    #     },
    #     {
    #         'noise_level': 0.005,
    #         'noise_length': 0.5,
    #         'noise_type': 'white_noise'
    #     },
    #
    #
    #     {
    #         'noise_level': 0.005,
    #         'noise_length': 0.1,
    #         'noise_type': 'gaussian_noise'
    #     },
    #     {
    #         'noise_level': 0.05,
    #         'noise_length': 0.1,
    #         'noise_type': 'gaussian_noise'
    #     },
    #     {
    #         'noise_level': 0.005,
    #         'noise_length': 0.3,
    #         'noise_type': 'gaussian_noise'
    #     },
    #     {
    #         'noise_level': 0.005,
    #         'noise_length': 0.5,
    #         'noise_type': 'gaussian_noise'
    #     }
    # ]
    # print('噪声类型\t\t噪声等级\t\t噪声长度\t\t差异性')
    # for test in noise_test:
    #     noise_diff = add_noise(image_path, mode3_audio_path, image_size, mode3_window_size, width_height=w_h,
    #                            mode=mode3, noise_level=test['noise_level'], noise_length=test['noise_length'], noise_mode=test['noise_type'])
    #
    #     print('{}\t\t{}\t\t{:.2f}%\t\t{:.2f}%'.format(test['noise_type'], test['noise_level'], test['noise_length']*100, noise_diff*100))

    # 误码率分析
    bits_error_rate_analysis(audio_path)
