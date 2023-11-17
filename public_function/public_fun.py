from PIL import Image, ImageOps
import numpy as np

# 比较两个比特流的差异性
def diff_with_two_bytes(bytes1, bytes2, is_bites = True):
    if len(bytes1) != len(bytes2):
        raise ValueError("比特流长度不一样")

    total_length = len(bytes1)
    if is_bites:
        total_length *= 8
        differences = sum(bin(b1 ^ b2).count('1') for b1, b2 in zip(bytes1, bytes2))
    else:
        differences = sum(b1 != b2 for b1, b2 in zip(bytes1, bytes1))
    return 0 if differences == 0 else differences / total_length

# 比较两张图片的差异性，返回百分值
def diff_with_two_image(image1_path, image2_path, is_bits = True):
    """
    比较两张图片的差异性，即两张图片RGB的差异性，但是本代码中，并不会去纠结两张图大小与模式是否相同
    :param image1_path: 图片1的路径
    :param image2_path: 图片2的路径
    :param is_bits: 如果为True，那么比较的位级的差异，如果为False，那么比较的是字节级的差异
    :return:返回差异的比例
    """
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    image1_bytes = image1.tobytes()
    image2_bytes = image2.tobytes()
    return diff_with_two_bytes(image1_bytes, image2_bytes)






# 图片转array
def image2array(image_path, need_embedded_size = True):
    # 使用直接读取文件的方式
    # with open(image_path, 'rb') as f:
    #     ret = f.read()
    """
    图片转np.array，4个字节表示宽，4个字节表示高，后面的才是图片的RGBA模式的
    :param image_path: 图片路径，类型是str，并没有做类型检查
    :return: 返回np.array
    """
    # 使用RGB排列的方式
    image = Image.open(image_path)
    if not image.mode == 'L':
        raise ValueError("要求输入图片是灰度（图片必须以JPEG的形式保存的灰度）")
    ret = image.tobytes()

    if not need_embedded_size:
        return np.frombuffer(ret, dtype=np.uint8)

    # 宽4个字节来表示，大端序，高4个字节来表示，大端序
    width = image.width.to_bytes(4, 'big')
    height = image.height.to_bytes(4, 'big')
    return np.frombuffer(width + height + ret, dtype=np.uint8)


# 字节流转图片
def bytes_save_image(bytes_stream, output_path, image_mode='L', width_height=None):

    # 使用直接写入的方式
    # with open(output_path, 'wb') as f:
    # f.write(bytes_stream)

    """
    从字节流中保存图片
    :param bytes_stream: 字节流，要求类型是bytes，并没有做类型检查
    :param output_path: 输出路径，要求类型是str，并没有做类型检查
    :param image_mode: 图片的模式，默认是JPEG，暂且不支持其他模式
    :return: None
    """

    # 使用RGB排列的方式
    if width_height:
        Image.frombytes(image_mode, width_height, bytes_stream).save(output_path)
    else:
        width = int.from_bytes(bytes_stream[0:4], 'big')
        heigth = int.from_bytes(bytes_stream[4:8], 'big')
        image_bytes = bytes_stream[8:]
        Image.frombytes(image_mode, (width, heigth), image_bytes).save(output_path)


def sort_by_energy_analyze(audio_array: np.ndarray, window_size: int):
    """
    :param audio_array: audio的np.array形式
    :param window_size: 能量分析的窗口大小，也就是一段能量由几帧构成
    :return: 返回能量段的排序下标
    """
    if not window_size:
        raise ValueError('能量分析中，没有指定能量窗口大小')
    index = range(0, len(audio_array), window_size)[:-1]

    energys = np.zeros(len(index))
    for j, i in enumerate(index):
        energys[j] = (np.sum(audio_array[i * window_size:(i + 1) * window_size] ** 2))
    return energys.argsort()[::-1]

def png_to_gray_image(input, output):
    image = Image.open(input)
    image = image.convert('L')
    image = ImageOps.invert(image)
    image.save(output)

