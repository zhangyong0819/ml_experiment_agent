---
doc_id: cluster-view_information
title: 查看信息
topic: cluster_view_information
error_type: RESOURCE_ERROR
source: university_cluster_manual
---

# 查看信息

## 查看队列

查看所有队列：`bqueues`

```sh
$ bqueues 
QUEUE_NAME      PRIO STATUS          MAX JL/U JL/P JL/H NJOBS  PEND   RUN  SUSP 
x7542!           50  Open:Active       -    -    -    -    24     0    24     0
e5645!           50  Open:Active       -    -    -    -     0     0     0     0
e52643tgb!       50  Open:Active       -    -    -    -     8     0     8     0
……
6226rib          30  Open:Active       -    -    -    -     0     0     0     0
5218             30  Open:Active       -    -    -    -     0     0     0     0
6230r            30  Open:Active       -    -    -    -    32     0    32     0
```

- `QUEUE_NAME`：队列名称
- `PRIO`：队列优先级，越大优先级越高
- `STATUS`：队列状态。Open/Closed 表示是否可以提交，即用户是否可以提交作业到该队列；Active/Inact 表示否可以派发，即该队列的作业是否会被分发到计算节点运行。Open:Active 表示可提交可派发，Open:Inact 表示可提交但是不派发。
- `NJOBS`：排队、运行和挂起的作业所占总CPU核数
- `PEND`：排队中的作业所需总CPU核数
- `RUN`：运行中的作业所占总CPU核数
- `SUSP`：挂起的作业所占总CPU核数

查看队列详细信息：`bqueues -l`

```sh
$ bqueues -l e5v3ib

QUEUE: e5v3ib
  -- CPU: 2*E5-2680v3, RAM: 256GB/128GB, NET: 56Gb FDR InfiniBand

……

SCHEDULING POLICIES:  FAIRSHARE  EXCLUSIVE
FAIRSHARE_QUEUES:  e5v3ib e5v3ib! e7v4ib x5650ib 6140ib 62v100ib 722080tiib 72rtxib 7702ib ……
DISPATCH_ORDER:  QUEUE
USER_SHARES:  [root=, 999999] ……

SHARE_INFO_FOR: e5v3ib/
 USER/GROUP   SHARES  PRIORITY  STARTED  RESERVED  CPU_TIME  RUN_TIME   ADJUST  GPU_RUN_TIME
root=       999999  202255.328      0        0      8456.5     1542       0.000             0
……

USERS: all ~test/ 
HOSTS:  f01+10 f02+10 f03+10 f04+10 f05s+10 f05l/ 
RES_REQ:  span[ptile=24]
Maximum slot reservation time: 43200 seconds
```

## 查看节点

```sh

lshosts			#查看节点配置和资源

lshosts -gpu	#查看节点GPU配置和拓扑结构

lsload			#查看节点当前负载信息

lsload -gpu		#查看节点GPU整体负载

lsload -gpuload	#查看节点每个GPU负载

bhosts			#查看所有节点状态
```

## 自动关机

集群会对动力环境进行监控，遇市电中断或温度过高，将会自动终止所有作业，按照安全顺序进行关机操作。

关机时会在 `/fs00/reports/bjobs/` 目录下会自动保存一份作业列表备查。如文件 /fs00/reports/bjobs/bjobs.20130728070457 表明2013年07月28日07点04分57秒时刻所有作业的状态（`bjobs -uall -w` 的输出），同时也说明这个时间点开始自动关机。

如果温度未触及高点，但已明显增高，为了防止温度继续增高，集群会停止派发新作业，并且关闭空闲节点。

## 进程监督

用户只能登录到登录节点，且可从登录节点登录到有自己正在运行作业的计算节点。

所有节点的用户进程都会被检查，如果发现如下情况，将会杀掉该节点此用户所有进程，并记录在 /fs00/reports/process 中

- 计算节点有进程未通过作业调度系统提交运行的
- 计算节点用户进程占用的CPU资源明显高于在作业调度系统中请求的CPU资源
- 登录节点用户占用大于2个CPU核心的