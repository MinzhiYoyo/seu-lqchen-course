import wave
import numpy as np
from public_function.public_fun import image2array, bytes_save_image, sort_by_energy_analyze, diff_with_two_image, \
    diff_with_two_bytes

np_type = np.int16

def window_embed(array, target: bool):
    """
    对窗口嵌入信息
    :param array:
    :param target:
    :return:
    """
    # size = len(array)
    # index_0 = np.where((array & 0x1) == 0)[0]
    # index_1 = np.where((array & 0x1) == 1)[0]
    # index_0_bysort = np.abs(array[index_0]).argsort()#[::-1]
    # index_1_bysort = np.abs(array[index_1]).argsort()#[::-1]
    # max_index = index_1 if len(index_1) > len(index_0) else index_0
    # min_index = index_1 if len(index_1) <= len(index_0) else index_0
    # random_num = len(max_index) - len(min_index)
    # random_num = random_num // 2 + 1
    # random_index = max_index[:random_num]
    # # 需要更改，
    # if (target and len(index_0) >= len(index_1)) or (not target and len(index_0) <= len(index_1)):
    #     array[random_index] ^= 1
    # return array

    array[len(array)//2] >>= 1
    array[len(array)//2] <<= 1
    array[len(array)//2] |= target

    return array

def window_extract(array) -> int:
    # num_0 = np.sum((array & 0x1) == 0)
    # # 根据0的个数和window_size来计算1的个数
    # num_1 = len(array) - num_0
    #
    # if num_0 > num_1:
    #     return 0
    # else:
    #     return 1
    return int(array[len(array)//2] & 0x1)

def embed_extract_bytes(audio_path: str, data: int | np.ndarray | bytes, save=False, window_size = None):
    """
    在embed_bytes中嵌入比特流，并且会自动调节能量窗口的大小。并且解析数据，然后计算差异率
    :param audio_path: 音频文件路径
    :param data: int或者np.ndarray，如果是int，那么表示随机生成data字节的比特流。如果是ndarray，那么就使用这个比特流作为嵌入数据
    :return:
    """

    with wave.open(audio_path, 'rb') as audio:
        params = audio.getparams()
        rate = audio.getframerate()  # 采样率
        frames_num = audio.getnframes()
        audio_array = np.frombuffer(audio.readframes(frames_num), dtype=np_type).copy()

    # 计算最大的window_size
    if isinstance(data, int):
        data = np.random.randint(0, 255, size=data, dtype=np.uint8)

    data_length = len(data)

    # 计算最大的window_size
    max_window_size = len(audio_array) // data_length // 8 - 1

    if (window_size and window_size > max_window_size) or (not window_size):
        window_size = max_window_size

    sort_index = sort_by_energy_analyze(audio_array, window_size)

    for i in range(data_length):
        for j in range(8):
            # 改成统计性，如果最低位0多，就是0，否则就是1
            # audio的区间是
            tmp_index = sort_index[i * 8 + j] * window_size
            audio_array[tmp_index: tmp_index + window_size] = window_embed(audio_array[tmp_index:tmp_index+window_size], (data[i] >> (7 - j)) & 0x1)

    # 是否需要保存
    if save:
        name = audio_path.split('.')[0] + '_' + str(data_length) + '.wav'
        with wave.open(name, 'wb') as audio_file:
            audio_file.setparams(params)
            audio_file.writeframes(audio_array)

    ############################## 提取信息  ###########################
    embed_audio_array = audio_array.copy()
    debug_sort = sort_index.copy()
    # 计算debug_sort与sort_index的差异
    sort_index = sort_by_energy_analyze(embed_audio_array, window_size)
    num_of_diff = np.sum(debug_sort != sort_index)
    extract_data = b''
    # debug_data = []
    for i in range(data_length):
        tmp = int(0)
        for j in range(8):
            tmp_index = sort_index[i * 8 + j] * window_size
            tmp <<= 1
            tmp |= window_extract(embed_audio_array[tmp_index: tmp_index + window_size])
        extract_data += tmp.to_bytes(1, 'big')
        # debug_data.append(tmp)

    return window_size, diff_with_two_bytes(data.tobytes(), extract_data)

def count_max_window_size_by_bytestream(audio_path, data_length):
    with wave.open(audio_path, 'rb') as audio:
        la = len(np.frombuffer(audio.readframes(audio.getnframes()), dtype=np_type))

    return la // data_length // 8 - 1

def count_max_window_size(audio_path, image_path, need_embed_size = False):
    with wave.open(audio_path, 'rb') as audio:
        la = len(np.frombuffer(audio.readframes(audio.getnframes()), dtype=np_type))

    lb = len(image2array(image_path, need_embed_size))
    return la // lb // 8

def embed_image(audio_path, image_path, output_path, window_size=None, embedded_mode='adaptive_LSB'):
    """
    嵌入图像，分为三种嵌入模式（要求嵌入与提取的模式要一致）
        第一种是简单的LSB，第二种是基于能量分析的自适应LSB，第三种是基于能量分析的自适应LSB但不需要传递排序序列（此种方式会有一定误码）
    :param audio_path:音频载体路径
    :param image_path:待隐藏图片路径
    :param output_path:输出音频路径
    :param window_size:窗口大小
    :param embedded_mode:嵌入模式，可选值：lsb（简单LSB）, adaptive_LSB（基于能量分析的自适应LSB）, adaptive_LSB_nosort（基于能量分析的自适应LSB但不需要传递排序序列）
    :return: 返回大小或者排序序列，取决于使用何种嵌入模式，第一、三种返回图片大小，第二种返回排序序列
    """
    if not embedded_mode in ['lsb', 'adaptive_LSB', 'adaptive_LSB_nosort']:
        raise ValueError("嵌入模式不正确，请看代码的注释")

    # 读取音频文件
    audio = wave.open(audio_path, 'rb')
    audio_frames = audio.readframes(audio.getnframes())
    audio_array = np.frombuffer(audio_frames, dtype=np_type).copy()
    audio.close()
    sort_index_by_energy = None
    if embedded_mode in ['adaptive_LSB', 'adaptive_LSB_nosort']:
        sort_index_by_energy = sort_by_energy_analyze(audio_array, window_size)

    # 读取图像文件
    need_embedded_size = True
    if embedded_mode == 'adaptive_LSB_nosort':
        need_embedded_size = False
    image_array = image2array(image_path, need_embedded_size=need_embedded_size)

    # 计算最大的window_size
    max_window_size = len(audio_array) // len(image_array) // 8
    if window_size and window_size > max_window_size:
        raise ValueError(
            "你设置的 window_size = {} 超过了能承受的最大 max_window_size = {}".format(window_size, max_window_size))
    else:
        window_size = max_window_size

    # 确保图像和音频大小匹配
    if len(audio_array) < len(image_array) * 8 + 8:
        raise ValueError("音频文件太短或者图片文件太大，请选择加长音频文件或者缩小图片文件的大小")

    # 嵌入图像数据到音频中
    for i in range(len(image_array)):
        for j in range(8):
            if embedded_mode == 'lsb':
                tmp_index = i * 8 + j
                audio_array[tmp_index] &= 0xFFFE
                audio_array[tmp_index] |= ((image_array[i] >> (7 - j)) & 0x1)
            elif (embedded_mode in ['adaptive_LSB', 'adaptive_LSB_nosort']):
                # 改成统计性，如果最低位0多，就是0，否则就是1
                # audio的区间是
                tmp_index = sort_index_by_energy[i * 8 + j] * window_size
                audio_array[tmp_index: tmp_index + window_size] = window_embed(audio_array[tmp_index:tmp_index+window_size], (image_array[i] >> (7 - j)) & 0x1)

    # 将嵌入图像的音频写入新文件
    with wave.open(output_path, 'wb') as embedded_audio:
        embedded_audio.setparams(audio.getparams())
        embedded_audio.writeframes(audio_array.tobytes())

    if embedded_mode in ['lsb', 'adaptive_LSB_nosort']:
        return len(image_array)
    elif embedded_mode == 'adaptive_LSB':
        return sort_index_by_energy[:len(image_array) * 8]


def extract_image(embedded_audio_path, output_image_path, sort_index_or_length=None, window_size=None,
                  extract_mode='adaptive_LSB', width_height=None):
    """
    提取模式，同嵌入模式
    :param embedded_audio_path: 嵌入了图像的音频路径
    :param output_image_path: 提取的图片输出路径
    :param sort_index_by_energy: 能量排序的序列或者图片长度（字节），取决于提取模式，第一、三种为图片长度，单位字节；第二种为排序序列；
    :param window_size: 窗口大小
    :param extract_mode: 同嵌入模式，看embedded_image那部分
    :return:
    """
    if extract_mode not in ['lsb', 'adaptive_LSB', 'adaptive_LSB_nosort']:
        raise ValueError("提取模式不正确，请看代码的注释")

    embedded_audio = wave.open(embedded_audio_path, 'rb')
    embedded_audio_frames = embedded_audio.readframes(embedded_audio.getnframes())
    embedded_audio_array = np.frombuffer(embedded_audio_frames, dtype=np_type)
    image_size = None

    if isinstance(sort_index_or_length, int):  # 如果是整数
        image_size = sort_index_or_length
        if extract_mode == 'adaptive_LSB_nosort':
            sort_index_or_length = sort_by_energy_analyze(embedded_audio_array, window_size)
    elif isinstance(sort_index_or_length, np.ndarray):
        image_size = len(sort_index_or_length) // 8

    # 提取图像
    if not image_size:
        raise ValueError("无法获取图片的尺寸")
    extract_image_data = b''
    for i in range(image_size):
        tmp = int(0)
        for j in range(8):
            if extract_mode == 'lsb':
                tmp_index = 8 * i + j
                tmp <<= 1
                tmp |= (int(embedded_audio_array[tmp_index]) & 0x1)
            elif extract_mode == 'adaptive_LSB' or extract_mode == 'adaptive_LSB_nosort':
                tmp_index = sort_index_or_length[i * 8 + j] * window_size
                tmp <<= 1
                tmp |= window_extract(embedded_audio_array[tmp_index: tmp_index + window_size])
        extract_image_data += tmp.to_bytes(1, 'big')

    # 保存图像
    bytes_save_image(extract_image_data, output_image_path, width_height=width_height)

# 用与embeded_extract_bytes相同的方法来嵌入字节流数据
def embed_bytes(audio_path: str, data, output_path: str, window_size=None):
    with wave.open(audio_path, 'rb') as audio:
        params = audio.getparams()
        rate = audio.getframerate()  # 采样率
        frames_num = audio.getnframes()
        audio_array = np.frombuffer(audio.readframes(frames_num), dtype=np_type).copy()

    # 计算最大的window_size
    if isinstance(data, int):
        data = np.random.randint(0, 255, size=data, dtype=np.uint8)

    data_length = len(data)

    # 计算最大的window_size
    max_window_size = len(audio_array) // data_length // 8 - 1

    if (window_size and window_size > max_window_size) or (not window_size):
        window_size = max_window_size

    sort_index = sort_by_energy_analyze(audio_array, window_size)

    for i in range(data_length):
        for j in range(8):
            # 改成统计性，如果最低位0多，就是0，否则就是1
            # audio的区间是
            tmp_index = sort_index[i * 8 + j] * window_size
            audio_array[tmp_index: tmp_index + window_size] = window_embed(audio_array[tmp_index:tmp_index+window_size], (data[i] >> (7 - j)) & 0x1)

    with wave.open(output_path, 'wb') as audio_file:
        audio_file.setparams(params)
        audio_file.writeframes(audio_array)

    return data_length



# 用与图片相同的方法来提取字节流数据，并返回字节流
def extract_bytes(embedded_audio_path, data_length, window_size=None, extract_mode='adaptive_LSB'):
    """
    :param embedded_audio_path:
    :param sort_index_or_length:
    :param window_size:
    :param extract_mode:
    :return:
    """
    with wave.open(embedded_audio_path, 'rb') as embedded_audio:
        embed_audio_array = np.frombuffer(embedded_audio.readframes(embedded_audio.getnframes()), dtype=np_type).copy()
    # 计算debug_sort与sort_index的差异
    sort_index = sort_by_energy_analyze(embed_audio_array, window_size)
    extract_data = b''
    # debug_data = []
    for i in range(data_length):
        tmp = int(0)
        for j in range(8):
            tmp_index = sort_index[i * 8 + j] * window_size
            tmp <<= 1
            tmp |= window_extract(embed_audio_array[tmp_index: tmp_index + window_size])
        extract_data += tmp.to_bytes(1, 'big')
        # debug_data.append(tmp)

    # 返回数据
    return extract_data



if __name__ == '__main__':
    # 隐藏字节流
    pass

    # image_path = 'seu.jpg'
    # gray_path = 'seu_gray.jpg'
    # output_path = 'seu_output.jpg'
    #
    # from public_function.public_fun import png_to_gray_image
    # from PIL import Image
    # png_to_gray_image(image_path, gray_path)
    # image = Image.open(gray_path)
    # w_h = image.size
    #
    # image_array = image2array(gray_path, False)
    #
    # bytes_save_image(image_array, output_path, width_height=w_h)
    #
    #
    #
    # print(diff_with_two_image(gray_path, output_path))

    # window_size = 2
    # audio_path = '启动.wav'
    # image_path = 'seu.png'
    #
    # # 模式1，lsb模式
    # mode1 = 'lsb'
    # mode1_audio_path = 'mode1_audio.wav'  # 带有隐藏信息的音频路径
    # mode1_extract_path = 'mode1_image.png'  # 提取出来的图片路径
    #
    # # 模式2，自适应lsb
    # mode2 = 'adaptive_LSB'
    # mode2_audio_path = 'mode2_audio.wav'  # 带有隐藏信息的音频路径
    # mode2_extract_path = 'mode2_image.png'  # 提取出来的图片路径
    #
    # # 模式3，自适应lsb，且不需要传递嵌入位置
    # mode3 = 'adaptive_LSB_nosort'
    # mode3_audio_path = 'mode3_audio.wav'  # 带有隐藏信息的音频路径
    # mode3_extract_path = 'mode3_image.png'  # 提取出来的图片路径
    # mode3_window_size = 99
    #
    # # 模式1，实验
    # image_size = embed_image(audio_path, image_path, mode1_audio_path, embedded_mode=mode1)
    # extract_image(mode1_audio_path, mode1_extract_path, image_size, extract_mode=mode1)
    #
    # # 模式2，实验
    # sort_index = embed_image(audio_path, image_path, mode2_audio_path, window_size, embedded_mode=mode2)
    # extract_image(mode2_audio_path, mode2_extract_path, sort_index, window_size, extract_mode=mode2)
    #
    # # 模式3，实验
    # image_size = embed_image('启动.wav', "seu_light.png", mode3_audio_path, mode3_window_size, embedded_mode=mode3)
    # extract_image(mode3_audio_path, mode3_extract_path, image_size, mode3_window_size, mode3, width_height=(25, 25))
    # # 比较两张图的差异性
    # print("源图像与目标图像的差异：{} %".format(100 * diff_with_two_image('seu_light.png', mode3_extract_path)))

    # # 隐藏文件
    # sort_index = embed_image(audio_path, image_path, energy_lsb_path, window_size)
    #
    # # 提取
    # extract_image(energy_lsb_path, energy_lsb_extract_path, sort_index, window_size)
