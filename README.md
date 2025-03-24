
## Installation

### 1. Install OS

Download [Ubuntu 20.04.4 LTS](https://releases.ubuntu.com/20.04/) (Focal Fossa) desktop version `.iso` file and install the OS. 

>Note: To avoid potential version conflicts of Python and Java on existing OS, we suggest installing a new Ubuntu virtual machine on VMM (e.g., [VirtualBox 5.2.24](https://www.virtualbox.org/wiki/Download_Old_Builds_5_2), VMware, etc.).

**Suggested system configurations:**\
(Our package can run on pure CPU environments)\
RAM: >2GB\
Disk: >30GB\
CPU: >1 core

### 2. Clone Source Code

Install `git` tool if you have not intalled it.

```shell
sudo apt install git
```

Download the project folder into the user's `HOME` directory.

```shell
cd ~
git clone 
```


### 3. Install Dependencies

```shell 
cd ~/SimSPI/
chmod +x install_dep.sh
./install_dep.sh
```

## How-to-Run

>Note: All commands are executed under the main project folder: `~/SimSPI/`.

### 1. Pre-processing

#### 1.1 Use Test Patch Samples

We provide 10 patch as test examples in `~/SimSPI/raw_patch/`.

To retrive their pre- and post-patch files, run the following commands under `~/SimSPI/`:

```shell
python3 get_ab_file.py nginx nginx 02cca547
python3 get_ab_file.py nginx nginx 661e4086
python3 get_ab_file.py nginx nginx 9a3ec202
python3 get_ab_file.py nginx nginx dac90a4b
python3 get_ab_file.py nginx nginx fc785b12
python3 get_ab_file.py nginx nginx 60a8ed26
python3 get_ab_file.py nginx nginx bd7dad5b
python3 get_ab_file.py nginx nginx 4c5a49ce
python3 get_ab_file.py nginx nginx 71eb19da
python3 get_ab_file.py nginx nginx 56f53316
```

In the above, the third column refers to the owner (i.e., `nginx`), the fourth column refers to the repository (i.e., `nginx`), and the last column refers to the commit ID (e.g., `02cca547`).

As the result, the pre-patch and post-patch files will be stored in `~/SimSPI/ab_file/`.

#### 1.2 Use Your Own Patches

You can pre-process your own patches by running:
```shell 
python3 get_ab_file.py [owner] [repository] [commitID]
```
where `[owner]`, `[repository]`, and `[commitID]` are the owner name, repository name, and commit ID of the patch hosted on GitHub.

### 2. Generate PatchCPGs

To generate PatchCPGs for all the patches processed by the last step, please run the following commands under `~/GraphSPD/`:

```shell 
chmod -R +x ./joern
sudo python3 gen_cpg.py
python3 merge_cpg.py
```

Here, `gen_cpg.py` will generate two CPGs for pre- and post-patch files, respectively.\
`merge_cpg.py` will generate a merged PatchCPG from the two CPGs.\
The output PatchCPGs will be saved in `~/GraphSPD/testdata/`.

### 3. Run PatchGNN

In `~/SimSPI/`, run the command:

```shell 
python3 test.py
```

The prediction results are saved in file `~/SimSPI/logs/test_results.txt`. 

See the results by running:

```shell 
cat logs/test_results.txt
```

The prediction results contains the PatchCPG file path and the predictions, where 1 represents security patch and 0 represents non-security patch.

```text
filename,prediction
./testdata/fc785b12/out_slim_ninf_noast_n1_w.log,0
./testdata/dac90a4b/out_slim_ninf_noast_n1_w.log,1
./testdata/60a8ed26/out_slim_ninf_noast_n1_w.log,1
./testdata/71eb19da/out_slim_ninf_noast_n1_w.log,0
./testdata/9a3ec202/out_slim_ninf_noast_n1_w.log,1
./testdata/bd7dad5b/out_slim_ninf_noast_n1_w.log,1
./testdata/661e4086/out_slim_ninf_noast_n1_w.log,1
./testdata/02cca547/out_slim_ninf_noast_n1_w.log,0
./testdata/4c5a49ce/out_slim_ninf_noast_n1_w.log,0
./testdata/56f53316/out_slim_ninf_noast_n1_w.log,0
```



