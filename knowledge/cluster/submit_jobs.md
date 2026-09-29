---
doc_id: cluster-submit_jobs
title: 提交作业
topic: cluster_submit_jobs
error_type: RESOURCE_ERROR
source: university_cluster_manual
---

# 提交作业

集群使用作业调度系统管理所有计算作业，该系统接受用户的作业请求，并将作业合理的分配到合适的节点上运行，因此所有用户均应通过作业调度系统提交计算作业，不可直接在任何节点上直接运行。

# LSF
用户使用`bsub`命令向作业调度系统提交作业，`bsub`选项非常繁多，可对作业进行非常细致的控制，这里简要介绍常用选项和方法。

## bsub 使用方式

### 命令行方式

```bsub [options] command [arguments]```

- `[options]` 为 bsub 的选项，可以设定队列、CPU核数等
- `command` 为计算程序，如果是 MPI 并行程序需要使用 mpirun 启动
- `[arguments]` 为计算程序的参数

> 例：提交一个作业到 e5v3ib 队列，需要24核的 MPI 并行程序
> ```sh
> $ bsub -q e5v3ib -n 24 "module load oneapi/2024.0/mpi && mpirun ./app"
> Job <3206000> is submitted to queue <e5v3ib>
> ```

### 脚本方式

```sh
bsub < jobfile
```

jobfile 为作业的 shell 脚本文件，文件名任意且不需要运行权限，脚本内容如下：
```sh
#BSUB [options]
command [arguments]
```
脚本中以`#BSUB`开头的行后跟`bsub`的选项，其它行为作业运行脚本

>例：提交一个作业到 e5v3ib 队列，需要48核，需要大内存节点，作业名为 MgSiO3，标准输出文件为`out`，标准错误输出文件为`err`，Intel MPI 的并行作业0
>
>```sh
>$ cat job.lsf 
>#BSUB -q e5v3ib
>#BSUB -n 48
>#BSUB -J MgSiO3
>#BSUB -o out
>#BSUB -e err
>module load ips/2018u4
>mpirun ./app
>
>$ bsub < job.lsf 
>Job <3207099> is submitted to queue <e5v3ib>.
>```
>
>等价如下命令行方式
>
>```sh
>$ bsub -q e5v3ib -n 48 -J MgSiO3 -o out -e err "module load ips/2018u4;mpirun ./app"
>Job <3207099> is submitted to queue <e5v3ib>.
>```
## bsub 常用选项
  - `-J job_name`：作业名称
##### 资源请求
  - `-n min_tasks[,max_tasks]`：作业需要CPU核数；例：需要四核 `-n 4`；需要4~8核均可 `-n 4,8` 
  - `-m`：作业运行的节点或节点组，多个节点写在双引号内并用空格分隔，节点组对应的具体节点可用 `bmgroup` 命令查看，此选项很复杂。在 hostname/hostgroup 前后可用这些符号：后加`!`指定头结点、后加`+[num]`指定节点使用顺序。例：指定在 c04n01 和 c04n02 运行`-m "c04n01 c04n02"`；指定可在 f01n01~n03，但是最希望在 f01n01、次希望在 f01n02`-m "f01n01+2 f01n02+1 f01n03"`；
  - `-R "res_req"`：资源请求串，此选项非常复杂；例：有的队列某些节点内存较大，需要大内存节点可以指定 -R largemem
  - `-R "select[hname!=host_name]"`：排除host_name节点，如果要排除多个节点中间用&&连接，如`-R "select[hname!=x001 && hname!=x002]"`排除x001和x002节点
  - `-x`：作业需要独占节点，无论申请多少核，作业均独占所运行的节点
  - `-W [hour:]minute`：作业运行最长时间，超过这个时间则被 kill
##### CPU绑定
  - `-R affinity[core:cpubind=core:membind=localprefer:distribute=pack]`：作业调度系统将进行CPU亲和性绑定，注意可能会和程序本身（如MPI）的绑定冲突，使用前请测试！
  
##### 自动重运行
  - `-r`：如果计算节点或系统故障则自动重新运行作业
  - `-Q "exit_code [exit_code ...]"`：根据作业退出码自动重新提交作业。使用空格分隔多个退出码，all指所有退出码，加`~`排除一个或多个退出码。
  
##### 输入输出
  - `-I`：交互式作业，可在作业运行期间和程序进行交互（如输入参数等），可在调试期间使用，正常计算请勿使用
  - `-K`：等待作业执行完才返回
  - `-i input_file`：标准输入文件
  - `-o output_file`：标准输出文件
  - `-e error_file`：标准错误输出文件
  - 以上三个选项的文件名中可以包含`%J`用于表示 JOBID。如果没有用`-o`或`-oo`指定标准输出文件，那么系统会自动设定为`output_%J`；如不想要输出文件请设置`-o /dev/null`

更多选项见[官方文档](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=reference-bsub)

## GPU 作业

提交作业时使用 `-gpu` 选项申请所需的 GPU 资源，计算进程只可见作业调度系统分配的 GPU。CPU 核自动按照申请节点 GPU 的比例分配，如一节点8个 GPU 和40个 CPU 核，申请2个 GPU 则分配10个 CPU 核。

`-gpu` 的各个选项用:分隔，默认值为`num=1:mode=shared:mps=no:j_exclusive=yes`，常用选项如下

- `num=number`：每台主机需要GPU的数量
- `mode=shared | exclusive_process`：GPU运行模式，`shared`对应 Nvidia/AMD DEFAULT compute mode、`exclusive_process`对应 Nvidia EXCLUSIVE_PROCESS
- `mps=yes | no`：开启或关闭Nvidia Multi-Process Service (MPS)。关闭MPS，多进程通过时间分片的方式共享GPU；开启MPS，多进程共享一个CUDA Context并发执行，增加了GPU利用率
- `aff=yes | no`：是否强制进行严格的 GPU-CPU 亲和性绑定，还需要配合`-R affinity[core:cpubind=core:membind=localprefer:distribute=pack]`才能一同完成GPU-CPU亲和性绑定
<!-- - `j_exclusive=yes`：指定分配的GPU为作业独享，**注意**：优先队列（以!结尾的队列）务必添加`j_exclusive=yes` -->

## 作业依赖
一个计算任务可能分成几步，而每一步对资源的需求不同，因此需要分开提交，但这些作业之间又具有依赖关系，bsub 可使用选项 -w 'dependency_expression'指定依赖关系。如果计算任务分成几步，但是每步对资源需求一样，那么请写在一个作业任务中依次执行。

- `-w 'done(job_ID | "job_name")'`：需要 job_ID 或 job_name 作业完成且状态为 DONE，即退出码为0
- `-w 'ended(job_ID | "job_name")`'：需要 job_ID 或 job_name 作业完成或退出，状态为 EXIT 或 DONE
- 支持逻辑表达式&& (AND)、|| (OR)、! (NOT)
- 孤儿作业（即依赖条件不可能满足的）1分钟后会被自动终止

更多详细信息见[官方文档](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=o-w-2)

## MPI/OpenMP 混合作业
OpenMP (Open Multi-Processing) 是一种共享内存方式的单进程多线程并行编程技术；MPI (Message Passing Interface) 是一种多进程基于信息传递的并行编程技术。OpenMP 的特点是单节点、进程内、多线程、基于共享内存的并行运算；MPI 的特点是单或多节点、进程间、非共享内存、基于消息传递的并行运算。

混合并行编程模型构建的应用程序可以同时使用 OpenMP 和 MPI ，节点内NUMA内进程内使用 OpenMP 共享内存并行可降低内存需求，跨节点跨NUMA跨进程使用 MPI 消息传递可大规模并行。需要注意的是，并不是一个节点一个MPI进程是最优的，这往往会导致跨NUMA的内存访问，因此需要通过测试确定最佳配比。

mpirun一般会根据环境变量LSB_MCPU_HOSTS启动相应的MPI进程，因此可以通过下列方法改变此环境变量中每个节点的CPU核数，以匹配MPI/OpenMP混合作业的MPI进程分布：

1. `#BSUB -n `指定的仍然是总CPU核数
2. 提交作业脚本中需要在计算命令前首先运行
	```sh
   source /fs00/software/lsf/misc/ompthreads.sh [N]
   ```
3. 每个 MPI 进程的 OpenMP 线程数量可以用环境变量`OMP_NUM_THREADS`指定或上述命令行参数指定，同时指定时命令行参数优先，需要保证每个节点的 CPU核数可以被线程数整除！

## 常用环境变量

### 作业运行时

- LSB_JOBID：作业ID
- LSB_QUEUE：队列名称
- LSB_JOBNAME：作业名称
- LSB_DJOB_NUMPROC：分配的CPU总核数
- LSB_DJOB_HOSTFILE：分配的节点列表文件，每行一个
- LSB_HOSTS：分配的节点列表，每个CPU核一个节点名的纯节点列表
- LSB_MCPU_HOSTS：分配的节点和核数列表，每个节点名和CPU核数的列表
```
LSB_DJOB_NUMPROC=6
LSB_HOSTS="node1 node1 node1 node2 node2 node2"
LSB_MCPU_HOSTS="node1 3 node2 3"
```
```
$ cat $LSB_DJOB_HOSTFILE
node1
node1
node1
node2
node2
node2
```
> LSB_HOSTS 和 LSB_MCPU_HOSTS 以不同的格式包含相同的信息，LSB_MCPU_HOSTS 比 LSB_HOSTS 更短更精简，如果 LSB_HOSTS 超过 4096 字节，则仅有 LSB_MCPU_HOSTS。


## 作业脚本示例

### 串行作业

　　提交一个串行作业到 e52660 队列，命令行方式和脚本方式分别为：

```sh
$ bsub -q e52660 ./app
Job <3279929> is submitted to queue <e52660>.
$ cat job.lsf 
#BSUB -q e52660
./app
```
```sh
$ bsub < job.lsf 
Job <3279930> is submitted to queue <e52660>.
```

### MPI 并行作业

　　**MPI程序需要使用`mpirun`启动**

　　提交一个需要48核的 Intel MPI 并行作业到 e5v3ib，命令行方式为：

```sh
$ bsub -q e5v3ib -n 48 "module load ips/2018u4;mpirun ./app"
Job <3280120> is submitted to queue <e5v3ib>.
```

　　提交一个需要48核的 Open MPI 并行作业到 e5v3ib，脚本方式为：

```sh
$ cat job.lsf 
#BSUB -q e5v3ib
#BSUB -n 48
module load iccifort/15.0.3 imkl/11.2.3 openmpi/1.10.0-iccifort-15.0.3
mpirun ./app

$ bsub < job.lsf 
Job <3280122> is submitted to queue <e5v3ib>.
```

###  OpenMP 并行作业

　　**OpenMP 不能跨节点，因此`-n`不能指定超过一个节点的CPU核数**

　　提交一个需要64核的 OpenMP 并行作业到 e7v4ib，使用程序参数 -nt 指定线程数量，命令行方式为：

```sh
$ bsub -q e7v4ib -n 64 "./app-nt \$LSB_DJOB_NUMPROC"
Job <3348175> is submitted to queue <e7v4ib>.
```

　　提交一个需要64核的 OpenMP 并行作业到 e7v4ib，使用环境变量`OMP_NUM_THREADS`指定线程数量，脚本方式为：

```sh
$ cat job.lsf 
#BSUB -q e7v4ib
#BSUB -n 64
OMP_NUM_THREADS="$LSB_DJOB_NUMPROC"
./app

$ bsub < job.lsf 
Job <3348182> is submitted to queue <e7v4ib>.
```

### MPI/OpenMP 混合作业
每个MPI进程跑6个OpenMP线程

通过环境变量`OMP_NUM_THREADS`指定

```sh
#BSUB -q 6140ib
#BSUB -n 72
export OMP_NUM_THREADS=6
source /fs00/software/lsf/misc/ompthreads.sh
module load ips/2018u4
mpirun ./run
```

 通过命令行参数指定（有些计算程序需要通过命令行参数指定线程数量）

```sh
#BSUB -q 6140ib
#BSUB -n 72
source /fs00/software/lsf/misc/ompthreads.sh 6
module load ips/2018u4
mpirun ./openmx -nt 6
```


### GPU 作业

提交一个需要1个 GPU 的作业到 e5v4p100ib 队列

```sh
bsub -q e5v4p100ib -gpu num=1 ./gpu_app
```

提交一个需要4个 GPU 的作业到 62v100ib 队列，进行 GPU-CPU 绑定

```sh
bsub -q 62v100ib -gpu "num=4:aff=yes" ./gpu_app
```

# Slurm
在 Slurm 中，`sbatch` 用于提交批处理作业；`srun` 用于启动作业步骤，通常用于运行并行程序，也可用于启动交互式任务。

## Slurm 作业提交方式

### 批处理作业方式 (sbatch) 

`sbatch` 用于提交批处理作业。用户将资源需求、环境配置和任务执行命令写入脚本，通过 `sbatch` 提交，由 Slurm 调度系统负责资源分配、作业调度和任务启动。

```sh
sbatch job.slurm
```
提交成功后，Slurm 返回作业 ID。


`sbatch` 直接读取脚本内容，因此 `job.slurm` 脚本文件不需要执行权限，其基本结构如下：
```sh
#!/bin/bash
#SBATCH [options]

command [arguments]
```
脚本中以 `#SBATCH` 开头的行用于向 Slurm 指定作业参数，包括资源需求、作业名称、输出文件、运行时间等。

> **示例：通过 `sbatch` 提交一个 MPI 并行作业**
>
> ```sh
> $ cat job.slurm 
> #!/bin/bash
> #SBATCH -p Partition         # 指定分区
> #SBATCH -J MyMPIJob          # 作业名称
> #SBATCH -N 2                 # 请求 2 个节点
> #SBATCH --ntasks-per-node=24 # 每个节点启动 24 个 task（MPI rank）
> #SBATCH -o output_%j.log     # 标准输出文件 (%j 会被替换为 Job ID)
> #SBATCH -e error_%j.log      # 标准错误文件
> 
> module load <MPI环境>
> srun ./app
> 
> $ sbatch job.slurm 
> Submitted batch job 3207099
> ```

### 交互式与单行命令方式 (srun)

`srun` 用于在分配到的资源上启动任务。配合 `--pty` 可以开启交互式终端，常用于代码调试：

```sh
# 申请 1 个节点、4 个核心，开启一个交互式 bash 终端
$ srun -p e5v3ib -N 1 -c 4 --pty bash
```

如果是临时提交一条简单的非交互命令，可以使用 `sbatch --wrap`：

```sh
$ sbatch -p e5v3ib -n 24 --wrap="srun ./app"
```

---

## 常用 sbatch/srun 选项

##### 基础选项

* `-p, --partition=partition_name`：指定作业提交的分区（队列）。
* `-J, --job-name=job_name`：指定作业名称。
* `-t, --time=[days-]hours:minutes:seconds`：作业运行的最长时间。超时会被系统自动 kill。

  * 示例：`-t 10` (10分钟), `-t 02:30:00` (2小时30分), `-t 1-00:00:00` (1天)。

##### 核心资源请求

在 Slurm 中，必须严格区分“任务 (Task/进程)”与“CPU核心 (CPU Core/线程)”。

* `-N, --nodes=num`：作业申请的节点总数。
* `-n, --ntasks=num`：作业申请的**总任务数**（通常对应 MPI 进程数）。如果不指定 `--cpus-per-task`，默认每个任务分配 1 个 CPU 核心。
* `--ntasks-per-node=num`：每个节点上运行的任务数。
* `-c, --cpus-per-task=num`：**每个任务需要的 CPU 核心数**（通常用于多线程程序，如 OpenMP）。作业请求的总 CPU 核数 = `ntasks` × `cpus-per-task`。
* `--mem=<size>[K|M|G|T]`：每个节点申请的内存大小（如 `--mem=16G`）。若未指定，系统会分配默认内存值，超出可能被 kill。
* `--mem-per-cpu=MB`：按每个 CPU 核心申请内存。
* 对于纯MPI作业，`cpus-per-task`默认为1；对于 OpenMP 或混合作业，必须显式设置-c。

##### 节点选择与控制

* `-w, --nodelist=host_list`：指定作业必须运行在哪些节点上，多个节点用逗号分隔（如 `-w "node01,node02"`）。
* `-x, --exclude=host_list`：排除发生故障或不希望运行的节点。
* `-C, --constraint=features`：请求具有特定属性（如大内存、特定 CPU 型号）的节点（如 `-C largemem`）。
* `--exclusive`：作业独占整个计算节点，即使只申请了部分核心，其他用户的作业也无法调度到该节点。

##### 输入输出控制

* `-o, --output=filename`：标准输出文件。可以使用 `%j` 代表作业 ID。若未显式指定，默认为 `slurm-%j.out`。
* `-e, --error=filename`：标准错误输出文件。
* `-W, --wait`：阻塞模式，等待作业执行完毕才返回终端提示符。

---
## 作业依赖

如果计算任务分多步进行，可以使用 `-d, --dependency=dependency_expression` 控制先后执行顺序。

* `-d afterok:job_ID`：当前作业需等待 `job_ID` 完成且**成功退出**（退出码为0）后才开始。
* `-d afternotok:job_ID`：当前作业需等待 `job_ID` 完成且**失败退出**（退出码非0）后才开始（常用于错误处理）。
* `-d afterany:job_ID`：无论 `job_ID` 成功与否，只要它运行结束，当前作业即开始。
* 支持多个依赖条件，用逗号 `,` 分隔或使用问号 `?`分割，但分隔符只能同时使用其中一种，不能混用。

---

## GPU 作业分配

申请 GPU 资源时，必须显式指定需要的 GPU 数量。**系统默认不会根据 GPU 数量自动为您分配相应比例的 CPU 或内存**，您应当手动搭配所需的 CPU 核心数。

* `--gres=gpu:number`：每个节点申请 `number` 个 GPU。
* `--gpus=number`：作业申请的总 GPU 数量。
* `--gpus-per-node=number`：每个节点申请的 GPU 数量。
  
示例：单节点 4 卡 GPU 作业

```sh
#!/bin/bash
#SBATCH -p gpu_part
#SBATCH -N 1
#SBATCH --gres=gpu:4        # 申请 4 张 GPU
#SBATCH -c 16               # 为这 4 张显卡显式搭配 16 个 CPU 核心处理数据
#SBATCH --mem=64G           # 显式申请内存

srun python train.py
```
* 若单个节点有多个型号GPU，需要指定GPU型号则使用
```bash
#SBATCH --gpus-per-node=<gpu_type>:<gpu_number>
```
示例：指定一块A100显卡
```bash
#SBATCH --gpus-per-node=a100:1
```
---