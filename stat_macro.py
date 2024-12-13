import os
import matplotlib.pyplot as plt
import math

testPath = "./raw_patch"

macroKeyWords = [
    '#if', '#ifdef', '#ifndef', '#elif', '#endif', '#else', '#define', '#undef'
]


def matchMacro(line):
    for keyWord in macroKeyWords:
        if keyWord.encode() in line:
            return True
    return False


def main():
    files = os.listdir(testPath)
    macro_count_list = []
    macro_ratio_list = []
    for file in files:
        # for file in ['567161fdd47aeb6987e700702f6bbfef04ae0236']:
        file_path = os.path.join(testPath, file)
        f = open(file_path, "rb")
        lines = f.readlines()
        f.close()
        line_count = 0
        macro_count = 0
        for line in lines:
            if line.startswith(b'+++') or line.startswith(b'---'):
                continue
            if not line.startswith(b'+') or line.startswith(b'-'):
                continue
            line = line.strip()
            if len(line) > 0:
                line_count += 1
            else:
                continue
            if matchMacro(line):
                macro_count += 1

        macro_count_list.append(macro_count)
        if line_count == 0:
            macro_ratio_list.append(0)
        else:
            macro_ratio_list.append(round(macro_count / line_count, 2))

    max_count = 30
    macro_count_array = [0] * max_count
    for macro_count in macro_count_list:
        if macro_count >= max_count:
            # macro_count_array[-1] += 1
            continue
        else:
            macro_count_array[macro_count] += 1

    max_ratio_count = 30
    macro_ratio_array = [0] * max_ratio_count
    for ratio in macro_ratio_list:
        idx = int(ratio / 0.01)
        if idx >= max_ratio_count:
            continue
        macro_ratio_array[idx] += 1

    print(macro_count_array)
    print(macro_ratio_array)
    plt.figure()
    plt.plot(range(1, len(macro_count_array)), macro_count_array[1:])
    plt.xticks(range(1, len(macro_count_array)))
    plt.savefig("img/macro_count.png", bbox_inches='tight')

    macro_ratio_xticks = [0.01 * k for k in range(1, max_ratio_count)]
    fig = plt.figure()
    plt.plot(macro_ratio_xticks, macro_ratio_array[1:])
    plt.xticks(macro_ratio_xticks, rotation=300)
    # fig.autofmt_xdate()
    plt.savefig("img/macro_ratio.png", bbox_inches='tight')


if __name__ == "__main__":
    main()
