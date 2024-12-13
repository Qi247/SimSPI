import os
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED


def get_all_commits_for_qemu():
    files = sorted(os.listdir("../mygraph/data/qemu/raw_patch"))

    with open("qemu.list", "w") as f:
        for file in files:
            f.write(os.path.basename(file) + "\n")


def get_patch(owner, repo, commit_id):
    download_url = "https://github.com/{:s}/{:s}/commit/{:s}.patch".format(
        owner, repo, commit_id)
    print(download_url)

    download_cmd = "wget -O raw_patch/{:s} {:s}".format(
        commit_id, download_url)
    print(download_cmd)
    os.system(download_cmd)


if __name__ == "__main__":
    # get_all_commits_for_qemu()

    f = open("qemu.list")
    commit_ids = f.readlines()
    f.close()

    executor = ThreadPoolExecutor(max_workers=4)
    tasks = []

    for commit_id in commit_ids[100:10000]:
        tasks.append(
            executor.submit(get_patch, "qemu", "qemu", commit_id.strip()))

    wait(tasks, return_when=ALL_COMPLETED)
