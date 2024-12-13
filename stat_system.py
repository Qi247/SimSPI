import os
import matplotlib.pyplot as plt

suffix_names = [
    'c', 'cpp', 'h', 'hpp', 'cc', 'java', 'php', 'go', 'js', 'py', 'pl',
    'json', 'rb', 'rs'
]


def main():
    patch_path = "./raw_patch"
    files = os.listdir(patch_path)
    line_count_list = []
    segment_count_list = []
    suffix_dict = {name: 0 for name in suffix_names}
    for file in files:
        line_count = 0
        segment_count = 0
        file_path = os.path.join(patch_path, file)
        with open(file_path, "rb") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(b"+") or line.startswith(b"-"):
                    line_count += 1
                if line.startswith(b"@@"):
                    segment_count += 1
                if line.startswith(b"---"):
                    suffix = line.split(b".")[-1].decode().strip()
                    if suffix in suffix_names:
                        suffix_dict[suffix] += 1

        line_count_list.append(line_count)
        segment_count_list.append(segment_count)

    # print(line_count_list, segment_count_list)

    line_count_array = [0] * 50
    for count in line_count_list:
        if count >= len(line_count_array):
            continue
        line_count_array[count] += 1
    print(line_count_array)
    s = ""
    for i in range(1, len(line_count_array) + 1):
        s += '{:d},'.format(i)
    print(s)

    segment_count_array = [0] * 20
    for count in segment_count_list:
        if count >= len(segment_count_array):
            continue
        segment_count_array[count] += 1
    print(segment_count_array)
    s = ""
    for i in range(1, len(segment_count_array) + 1):
        s += '{:d},'.format(i)
    print(s)

    # print(suffix_dict)
    ks = [
        'c', 'cpp', 'h', 'hpp', 'cc', 'java', 'php', 'go', 'js', 'py', 'pl',
        'json', 'rb', 'others'
    ]
    vs = [
        5602, 1033, 3746, 201, 124, 1096, 233, 12, 498, 262, 44, 363, 24, 2300
    ]
    new_vs = []
    for v in vs:
        new_vs.append(round(v / sum(vs), 2))
    print(ks)
    print(new_vs)

    new_ks = [
        'c', 'cpp', 'h', 'hpp', 'cc', 'java', 'php', 'go', 'js', 'py', 'others'
    ]
    vs = [5602, 1033, 3746, 201, 124, 1096, 233, 12, 498, 262, 123]
    new_vs = []
    for v in vs:
        new_vs.append(round(v / sum(vs), 2))
    print(new_ks)
    print(new_vs)


if __name__ == "__main__":
    main()
