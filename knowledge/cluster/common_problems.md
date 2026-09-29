---
doc_id: cluster-common_problems
title: 常见问题
topic: cluster_common_problems
error_type: UNKNOWN
source: university_cluster_manual
---

# 常见问题

# 网络
### 登录节点进行网络接入认证

#### 在登录节点登录校园网账号

集群中的登录节点进行网络接入认证（即登入南京大学校园网认证`p.nju.edu.cn`）后才可访问互联网，认证后登录节点**所有用户**均可访问互联网（相当于整个登录节点共享），请注意网络与账号隐私安全！

命令行登录和登出`p.nju.edu.cn`的方法有：
```sh
curl -X POST https://p.nju.edu.cn/api/portal/v1/login -H "Content-type: application/json" -d '{"username":"<username>","password":"<password>"}'
curl -X POST https://p.nju.edu.cn/api/portal/v1/logout -H "Content-type: application/json" -d '{}'
```

```sh
curl -s "http://p2.nju.edu.cn/portal_io/login?username=<username>&password=<password>"
curl -s http://p2.nju.edu.cn/portal_io/logout
```

**【强烈推荐】**如果有一些软件源需求，如`conda`、`pip`等包管理，或者需要外部代码库、容器，您可以通过中心提供的下列服务直接在集群上使用，无需登录校园网认证：

+ `mirror.nju.edu.cn`：`conda`、`pip`等大量常用软件源或代码仓库镜像
+ [代码托管](https://doc.nju.edu.cn/books/16790/page/git)：可手动镜像各大在线git仓库作为中转
+ [私服仓库](https://doc.nju.edu.cn/books/357a6)：各软件源、容器缓存

### 计算节点访问网络

所有计算节点均不能自由无限制的访问外网（含校园网和互联网），如需访问可以单独申请开放。当前已经放开的网络访问
<iframe className="dtable-embed" src="https://table.nju.edu.cn/dtable/view-external-links/custom/hpc-internet/" frameBorder="0" width="100%" height="1000" style="background: transparent; border: 1px solid #ccc;"></iframe>


### SSH/SFTP登录时报 no matching host key type found
SSH/SFTP登录时报错`no matching host key type found. Their offer: ssh-rsa,ssh-dss`，命令行添加选项`-o HostKeyAlgorithms=+ssh-rsa`或在`~/.ssh/config`中增加`HostKeyAlgorithms +ssh-rsa`


# 安装
### pip安装包到自己的目录下
Python的大多数包不需要root权限也能安装，只需在pip install后加-t指定安装目录即可，如：
```sh
cd scikit-opt-master
pip install -t $HOME .
```
这样就装到自己的家目录下。在~/.bashrc里或者作业脚本中加上环境变量
```sh
export PYTHONPATH=$HOME:$PYTHONPATH
```

# 作业
### 作业运行时实际占用CPU核数过多

1. 在作业中限定使用的CPU核数与申请核数相同，否则超过申请核数使用资源的作业会被杀掉。

特别是MATLAB/Python，很多Python包会自动多核并行计算，需要使用环境变量（如 `OMP_NUM_THREADS`、`NUMEXPR_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`）等方式设定线程数，与进程数一起匹配申请的核数。
```
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC
export NUMEXPR_NUM_THREADS=$LSB_DJOB_NUMPROC
export OPENBLAS_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_NUM_THREADS=$LSB_DJOB_NUMPROC
```

2. 可以尝试在提交作业时添加如下参数，做CPU亲和绑定
```
#BSUB -R affinity[core:cpubind=core:membind=localprefer:distribute=pack]
```
3. 如果实在不行可以 `#BSUB -x` 独占节点运行作业

### 排队作业数上限
动态限制用户排队作业数不能超过MAX(1000,MIN((30000-总作业数)/10,(20000-总排队作业数)/6))

即用户排队作业数，不超过三万减去总作业数的十分之一，也不超过二万减去总排队作业数的六分之一，但保底一千个。

### 作业状态UNKWN

一般是两种情况之一，如果作业状态长时间处于UNKWN状态，一般来说就是计算节点坏了可以直接杀掉作业。
1. 因为计算节点负载过高，未能及时获取作业状态导致状态未知，这种情况一般只需要等待即可，待负载下降获取状态后就正常了。
2. 因为计算节点出现故障且长时间未恢复，调度系统无法获取作业状态，此时如果登录不到相应的计算节点，可以直接杀掉作业。

# 路径
### $HOME相对路径
让/bbfs相应目录保持在$HOME下的相对路径，这样可以轻松应对未来路径的变更
```
[<username>@login1 ~]$ ln -s /bbfs$HOME cache
[<username>@login1 ~]$ ln -s /bbfs/scratch/$USER scratch
[<username>@login1 ~]$ ls -l|grep bbfs
lrwxrwxrwx 1 <username> <username> 26 Mar 16 11:45 cache -> /bbfs/<home-root>/<username>/<username>
lrwxrwxrwx 1 <username> <username> 20 Mar 16 11:45 scratch -> /bbfs/scratch/<username>
```

# 连接
### VSCode 远程连接集群
VSCode 的 Remote-SSH 插件是远程开发的利器，但由于集群启用了两步认证（2FA），且登录节点（entry）不支持隧道转发，因此无法直接通过常规方式连接。解决方案是：**在节点上启动一个用户态 SSH 服务，绑定一个高位端口，然后使用公钥直连该端口**。

选择的节点必须得有校园网公网ip，如210.28.xxx.xxx，建议使用login节点。

#### 1、在节点上配置用户态 SSH 服务

1. 创建服务目录并生成主机密钥
   ```bash
   mkdir -p ~/.sshd
   ssh-keygen -t rsa -f ~/.sshd/ssh_host_rsa_key -N ""
   ```

2. 编写配置文件 `~/.sshd/sshd_config`
   ```bash
   vim ~/.sshd/sshd_config
   ```
   写入以下内容（端口可自定义，此处以 `2222` 为例）：
   ```conf
   Port 2222
   ListenAddress 0.0.0.0
   HostKey ~/.sshd/ssh_host_rsa_key
   PasswordAuthentication no
   ChallengeResponseAuthentication no
   UsePAM yes
   PrintMotd no
   Subsystem sftp internal-sftp
   ```

3. 配置公钥登录（本地公钥写入服务器）
   - 在本地（你的电脑）获取公钥内容。Windows PowerShell 中执行：
     ```powershell
     type $env:USERPROFILE\.ssh\id_rsa.pub
     ```
     若使用 ED25519 则替换为 `id_ed25519.pub`。
   - 若本地无公钥则新建。Windows PowerShell 中执行：
     ```powershell
     ssh-keygen -t rsa -b 4096 -C "你的邮箱@example.com"
     ```
     若使用 ED25519 则在 Windows PowerShell 中执行：
     ```powershell
     ssh-keygen -t rsa -b 4096 -C "你的邮箱@example.com"
     ```
     执行后会提示你保存文件的位置，直接按回车使用默认路径。接着提示设置密码短语（passphrase），按回车留空（不设密码）。然后就可以执行获取公钥内容的步骤。
   - 回到节点，将公钥写入 `~/.ssh/authorized_keys`：
     ```bash
     mkdir -p ~/.ssh
     chmod 700 ~/.ssh
     vim ~/.ssh/authorized_keys   # 粘贴公钥内容，保存退出
     chmod 600 ~/.ssh/authorized_keys
     chmod 755 ~                  # 修复主目录权限，避免认证失败
     ```

4. **启动用户态 sshd（务必使用非 debug 模式）**
   ```bash
   /usr/sbin/sshd -f ~/.sshd/sshd_config
   ```
   验证监听端口是否成功：
   ```bash
   ss -tlnp | grep 2222
   ```
   应看到类似 `0.0.0.0:2222` 的输出。

#### 2、本地连接测试

在本地终端中测试 SSH 连接（替换 IP 为节点的校园网 IP，可通过 `hostname -I` 查看）：
```bash
ssh -p 2222 用户名@210.28.xxx.xxx
```
若测试成功，再测试 SFTP（VSCode 依赖）：
```bash
sftp -P 2222 用户名@210.28.xxx.xxx
```
正常进入 sftp 交互界面即表示一切就绪。

#### 3、配置 VSCode

1. 编辑本地 SSH 配置文件（路径 `~/.ssh/config`），加入主机条目：
   ```ssh
   Host cluster-dev
       HostName 210.28.xxx.xxx
       User 你的集群用户名
       Port 2222
       IdentityFile ~/.ssh/id_rsa   # 或 id_ed25519，与本地公钥对应
   ```

2. 在 VSCode 中连接
   - 安装插件 Remote - SSH（如未安装）。
   - 按下 `F1` 或点击左下角齿轮图标，选择 Remote-SSH: Connect to Host...，然后输入 `cluster-dev`。
   - 首次连接时，VSCode 会自动在计算节点上安装 `.vscode-server`，稍等片刻即可。
   - 连接成功后，打开远程文件夹（如你的作业目录），即可像本地一样编辑、调试和运行代码。

> **注意**：若 VSCode 连接后打开文件夹时断开，通常是因为 sshd 以 debug 模式运行（`-d` 参数）导致的，请确保启动命令不包含 `-d`。若出现该问题，先 `pkill -u $USER sshd` 结束所有用户态 sshd，再重新用正确命令启动。

#### 4、进程管理（可选）

- 查看当前用户启动的 sshd：
  ```bash
  ps -ef | grep sshd | grep $USER
  ```
- 关闭所有用户态 sshd：
  ```bash
  pkill -u $USER sshd
  ```

> **安全提醒**：该直连方式仅在校园网或 VPN 环境下使用，切勿暴露至公网；建议关闭密码登录（仅保留公钥）以防范暴力破解。

#### 5、备选方案：网页版 VSCode（code-server）

若 SSH 方案不稳定，也可在计算节点上部署 code-server，通过浏览器访问：

```bash
# 需先安装或使用模块
code-server --bind-addr 0.0.0.0:8080
```
然后在本地浏览器访问 `http://210.28.xxx.xxx:8080` 即可获得浏览器版 VSCode 界面（注意密码保护，建议设置 `--auth password`）。