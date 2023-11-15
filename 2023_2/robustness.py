from audio_hiding import embed_extract_bytes
import tqdm
# 分析误码率
def bits_error_rate_analysis():
    data_size = []
    error_rate = []
    window_size = []
    output_str = "数据长度/字节\t\t误码率/比特级\t\t窗口大小\n"
    for d in tqdm.tqdm(range(1000, 102400, 500)):
        winsize, rate = embed_extract_bytes('启动_big.wav', d, False)
        data_size.append(d)
        error_rate.append(rate)
        window_size.append(winsize)
        output_str += str(d) + '\t\t' + str(rate) + '\t\t' + str(winsize) + '\n'
    with open('误码率结果.txt', 'w') as f:
        f.write(output_str)


# 误码率分析
# bits_error_rate_analysis()


