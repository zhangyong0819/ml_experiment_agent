---
doc_id: cluster-integrated_cloud
title: 集成云盘
topic: cluster_integrated_cloud
error_type: UNKNOWN
source: university_cluster_manual
---

# 集成云盘

### 独立账号

高性能计算集群用户可以使用集群账号直接登录云盘，用户名为`集群账号@hpc.nju.edu.cn`，密码即集群账号密码；如集群账号为`<username>`，用户名填写`<username>@hpc.nju.edu.cn`。首次使用需要登录网页端激活账号，但集群账号不能使用云盘的上下传外链功能。

云盘上的集群账号与南京大学统一身份认证账号相互独立，如集群账号`<username>@hpc.nju.edu.cn`和南京大学统一身份认证账号`0102003`是两个完全独立的账号，需要分别激活才能使用。

### 文件同步

通过云盘的多平台多终端同步功能，可将集群中的目录和本地计算机的目录同步，对本地目录的操作几乎立刻反应在集群的目录中，不再需要通过 SFTP 上下传文件。

- 计算结果输出到集群同步目录中，本地计算机会自动下载，可在本地直接查看和编辑输出文件；
- 计算输入文件放到本地同步目录中，集群会自动下载，直接引用输入文件即可提交作业。

### 客户端

登录节点已安装云盘同步客户端和挂载盘客户端

- 同步客户端命令行界面：seaf-cli（[Linux命令行同步客户端手册](https://help.seafile.com/syncing_client/linux-cli/)）
- 同步客户端图形化界面：seafile-applet
- 挂载盘客户端命令行界面：seadrive（[Linux命令行挂载盘手册](https://help.seafile.com/drive_client/drive_client_for_linux/#running-seadrive-without-gui)）
- 挂载盘客户端图形化界面：seadrive-gui

除官方客户端外，也支持`rclone`便捷操作。

#### 挂载盘命令行客户端

1. 获取Token（`<username>`替换为集群账号，`<password>`替换为密码）
   ```sh
   curl -d 'username=<username>@hpc.nju.edu.cn' -d 'password=<password>' https://box.nju.edu.cn/api2/auth-token/
   ```

2. 创建配置文件`~/seadrive.conf`，`<username>`替换为集群账号，`<token>`替换为上一步的返回值
   ```
   [account]
   server = https://box.nju.edu.cn
   username = <username>@hpc.nju.edu.cn
   token = <token>
   is_pro = true
   [general]
   client_name = hpc-login
   [cache]
   size_limit = 10GB
   clean_cache_interval = 10
   ```
3. 启动挂载盘，命令在前台运行，新开一个窗口进行其它操作
   ```sh
   seadrive -c ~/seadrive.conf -f -d ~/.seadrive/data -l ~/.seadrive/data/logs/seadrive.log ~/SeaDrive
   ```
4. 如果报错 `Transport endpoint is not connected` ，执行：`fusermount -u ~/SeaDrive`
5. `~/SeaDrive` 目录就是云盘挂载在本地的目录，可以直接读写访问

#### 同步命令行客户端

1. 创建客户端本地数据目录（目录名可更改）
   ```sh
   mkdir ~/Seafile
   ```
2. 初始化客户端并指定本地数据目录（上一步创建的目录）
   ```sh
   seaf-cli init -d ~/Seafile
   ```
3. 启动客户端
   ```sh
   seaf-cli start
   ```
4. 列出云盘资料库ID（`<username>`替换为集群账号）
   ```sh
   seaf-cli list-remote -s https://box.nju.edu.cn -u <username>@hpc.nju.edu.cn
   ```
5. 将本地目录与云盘资料库同步（`<library-id>`替换为上一步列出的某个资料库ID，`<folder>`替换为本地目录）
   ```sh
   seaf-cli sync -s https://box.nju.edu.cn -u <username>@hpc.nju.edu.cn -l <library-id> -d <folder>
   ```
	将云盘资料库下载到本地目录
   ```sh
   seaf-cli download -s https://box.nju.edu.cn -u <username>@hpc.nju.edu.cn -l <library-id> -d <folder>
   ```
6. 查看当前同步状态
   ```sh
   seaf-cli status
   ```
7. 解除本地目录同步
   ```sh
   seaf-cli desync -d <folder>
   ```
8. 停止客户端
   ```sh
   seaf-cli stop
   ```

#### 使用 rclone 进行云端文件操作

[rclone](https://rclone.org/) 是一个强大的云存储管理工具，云盘也在其支持的存储类型内。使用`rclone`可以更加符合命令行用户使用习惯地去存取云盘文件。

1. （登录后第一次需要）启用`rclone`工具
   ```sh
   module load rclone
   ```
2. （仅需设置一次）初次使用`rclone`需要进行设置。在`~/.config/rclone/`目录下创建一名为`rclone.conf`的文件，内容（`<username>`替换为集群账号）如下：
   ```ini
   ;不一定必须叫 hpc_box，你可以任意设置这个标签，但是后文必须相应地使用该标签
   [hpc_box]
   type = seafile
   url = https://box.nju.edu.cn
   user = <username>@hpc.nju.edu.cn
   ;如果开启了两步认证，下条设置为true
   2fa = false
   ```
3. （仅需设置一次）设置密码。`<password>`替换为密码：
   ```sh
   rclone config update hpc_box pass <password>
   ```
4. 设置完毕后，之后按照下方使用即可。列出云端目录：
   ```sh
   rclone lsd hpc_box:
   ```
5. 创建云端资料库（`<library>`替换为资料库名）：
   ```sh
   rclone mkdir hpc_box:<library> --seafile-create-library
   #例如， rclone mkdir hpc_box:test --seafile-create-library 将在云盘创建一个名为`test`的资料库。
   ```
6. 向云端上传文件。`<destination_path>`替换为需要上传的本地路径，`<source_path>`替换为云端路径：
   ```sh
   rclone copy <destination_path> hpc_box:<source_path>
   
   #例如，假设本地当前目录下有个 myfile.txt 文件：
   #rclone copy myfile.txt hpc_box:test 将把本地当前目录的 myfile.txt 文件上传至云端的 test 资料库下。
   #再例如，假设本地当前目录下有个 myfolder 文件夹，里面有很多文件：
   #rclone copy myfoler hpc_box:test 将把本地当前目录的 myfolder 文件夹内的所有文件夹和文件（不包括 myfolder 自身）上传至云端的 test 资料库下。
7. 从云端下载文件：
   ```sh
   rclone copy hpc_box:<source_path> <destination_path>
   
   #例如，假设云端资料库`test`下有个 myfile.txt 文件：
   #rclone copy hpc_box:test/myfile.txt . 将把云端资料库`test`下的 myfile.txt 下载至本地当前目录。
   #例如，假设云端资料库`test`下有个 myfolder 文件夹，里面有很多文件：
   #rclone copy hpc_box:test/myfolder . 将把云端资料库`test`下的 myfolder 文件夹内的所有文件夹和文件（不包括 myfolder 自身）下载至本地当前目录。
8. 在传输过程中，如果加上`-P`会显示进度条，强烈建议添加；如果上传或下载过程中，文件较多，建议再加上`--no-traverse`避免列出文件导致传输时间过长：
   ```sh
   rclone copy hpc_box:<source_path> <destination_path> -P --no-traverse
   
   #例如，假设云端资料库`test`下有个 myfile.txt 文件：
   #rclone copy hpc_box:test/myfolder . -P --no-traverse 将把云端资料库`test`下的 myfolder 文件夹内的所有文件夹和文件（不包括 myfolder 自身）下载至本地当前目录，且显示进度条，不列出所有文件。
   ```
9. 更多`rclone`的用法见[rclone - commands](https://rclone.org/commands/)；若有加密资料库等其他云盘需求，或查看`rclone`对云盘的支持功能，请参考[rclone - seafile](https://rclone.org/seafile/)或[rclone - storage systems](https://rclone.org/overview/)。