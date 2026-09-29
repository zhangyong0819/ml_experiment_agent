---
doc_id: cluster-containerization
title: 容器化
topic: cluster_containerization
error_type: UNKNOWN
source: university_cluster_manual
---

# 容器化

## 容器技术
容器技术能够对应用及其整个运行时环境（包括全部所需文件）一起进行打包或隔离。从而可以在不同环境（如开发、测试和生产等环境）之间轻松迁移应用，同时还可保留应用的全部功能。有了容器不再需要管理员为你安装任何东西。容器是一个独立的系统，你可以在里面做任何想做的事情。

## Apptainer
2021年11月，Singularity 开源项目加入 Linux 基金会，并更名为 Apptainer。Apptainer 是一套类似于 Docker 的容器解决方案，是用于HPC容器系统。Apptainer 兼容 Singularity Image File (SIF)、Singularity 的环境变量、Singularity 的命令、自动迁移 Singularity 的用户配置。

### 镜像
一般保存为压缩只读Singularity Image File (SIF)格式的镜像文件。

#### 公共镜像
在 `/fs00/software/singularity-images/` 已存放了许多常用的SIF镜像文件，可直接使用。

#### 导入镜像
以从 Docker Hub 等镜像仓库或镜像文件直接创建SIF镜像文件
- 从 Docker Hub 导入镜像：`apptainer build ubuntu.sif docker://docker.nju.edu.cn/library/ubuntu`
- 从 NVIDIA NGC 导入镜像：`apptainer build ngc_cuda.sif docker://ngc.nju.edu.cn/nvidia/cuda`
- 从 docker save 保存的镜像文件导入镜像：`apptainer build abc.sif docker-archive://abc.tar`

#### 制作镜像
1. 创建沙盒目录：`apptainer build --fix-perms --sandbox build docker://docker.nju.edu.cn/library/ubuntu`
2. 进入沙盒容器：`apptainer shell build/`
3. 在容器中安装和编译软件：如`apt` `make`
4. 退出容器：`exit`
5. 将沙盒打包成SIF镜像文件：`apptainer build abc.sif build`

因登录节点无root权限，可能出现问题，因此建议在自己的系统上制作镜像。

#### CI创建镜像
用 [git.nju.edu.cn](https://git.nju.edu.cn) 上的 CI/CD 自动调用 kaniko 构建 Docker 镜像，详见[CI/CD 使用指南](https://doc.nju.edu.cn/books/16790/page/cicd)，然后再导入`apptainer build ocr.sif docker://reg.nju.edu.cn/<username>/ocr`

一个简单的示例见：[CI/CD自动化构建Docker镜像](https://doc.nju.edu.cn/books/16790/page/docker)


### 提交作业

提交容器作业的示例脚本
```sh
#BSUB -q 62v100ib
#BSUB -gpu num=4

apptainer exec --nv cuda.sif app
```
### 常用选项

- `--nv`：提供NVIDIA GPUs & CUDA支持
- `--bind/-B src[:dest[:opts]]`：绑定额外的路径


### 相关网站

- [Apptainer](https://apptainer.org/)
- [Apptainer User Guide](https://apptainer.org/docs/user/latest/)
- [Docker Hub](https://hub.docker.com/)
- [NVIDIA NGC](https://ngc.nvidia.com/)