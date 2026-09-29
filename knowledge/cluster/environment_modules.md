---
doc_id: cluster-environment_modules
title: 环境变量
topic: cluster_environment_modules
error_type: UNKNOWN
source: university_cluster_manual
---

# 环境变量

集群使用Environment Modules以模块的形式对环境变量进行管理。在高性能计算集群系统中，安装有多种软件及其不同版本，它们需要设置不同的环境变量，Environment Module可以将这些环境变量做成模块文件（modulefile）。模块可以被加载(load)、卸载(unload)、切换(switch)等，这些操作会改变相应的环境变量设置，让用户方便的在不同环境之间切换。相比将环境变量设置写入/etc/profile、\~/.bashrc或\~/.bash_profile，Environment Module操作只影响当前用户的当前登录环境；相比直接source文件，Environment Module的操作可以撤销（卸载）。普通用户还可以自己编写module，具有很好的定制性。用户不仅可以在命令行中Environment Modules，也能在作业提交脚本中使用，对编译和计算环境都能够很好的控制。

### 常用命令

- 显示module帮助：module help
- 显示所有可用模块：module avail
- 显示已加载模块：module list
- 加载模块：module load MODULEFILE
- 卸载模块：module unload MODULEFILE
- 切换模块：module switch OLD_MODULEFILE NEW_MODULEFILE（等价于：module unload OLD_MODULEFILE; module load NEW_MODULEFILE）
- 卸载所有已加载模块：module purge
- 显示模块说明：module whatis MODULEFILE
- 显示模块内容：module display MODULEFILE
- 增加搜索目录：module use
- 减少搜索目录：module unuse

### 非 Bash

使用非 Bash 作为 Login Shell 的用户，请在作业提交脚本的首行指明 Shell 或使用 bsub 选项 -L 指定作业运行时 Login Shell 为 Bash

如 Login Shell 为 tcsh，使用下面两种方法中的一种

```sh
#!/bin/tcsh #首行指明Shell
#BSUB ...
...
```
```sh
...
#BSUB -L /bin/bash #指定运行时Login Shell为Bash
...
```

### 加载模块示例

```sh
$ module avail #查看所有可用模块

------------------------------ /fs00/software/modulefiles ------------------------------
gcc/5.2.0                          impi/5.0.3.048
iccifort/15.0.3                   ips/2011sp1u3
ics/2013                           ips/2015u3
ics/2013sp1u1                  openmpi/1.10.0-gcc-5.2.0
imkl/11.2.3                      openmpi/1.10.0-iccifort-15.0.3

$ module list #显示已加载模块
No Modulefiles Currently Loaded. #没有模块被加载

$ icc --version #故icc找不到
-bash: icc: command not found

$ module whatis ips/2015u3 #查看模块说明
ips/2015u3           : Intel Parallel Studio XE 2015 Update 3 Cluster Edition

$ module load ips/2015u3 #加载模块
$ icc --version
icc (ICC) 15.0.3 20150407
Copyright (C) 1985-2015 Intel Corporation.  All rights reserved.

$ module list #显示已加载模块
Currently Loaded Modulefiles:
  1) ips/2015u3

$ module unload ips/2015u3 #卸载模块
```

### 搜索路径示例

```sh
$ module use
Search path for module files (in search order):
  /fs00/software/modulefiles

$ module avail 
--- /fs00/software/modulefiles ---
gcc/12.1.0
……                                


$ module use /fs00/software/modulefiles/oneapi/2024.0 #添加MODULEPATH搜索目录

$ module use
Search path for module files (in search order):
  /fs00/software/modulefiles/oneapi/2024.0
  /fs00/software/modulefiles

$ module avail 
--- /fs00/software/modulefiles/oneapi/2024.0 ---
mkl32/2024.0
compiler/2024.0.2
mkl/2024.0
……

--- /fs00/software/modulefiles ---
gcc/12.1.0
……
```