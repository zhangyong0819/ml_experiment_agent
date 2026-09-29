---
doc_id: cluster-remote_login
title: 远程登录
topic: cluster_remote_login
error_type: UNKNOWN
source: university_cluster_manual
---

# 远程登录

## 登录方式

### [远程控制-登录方式](https://doc.nju.edu.cn/books/37693/page/a5bfc)

请务必**首先仔细阅读**上述**远程控制-登录方式**内容，并**合并覆盖以下**内容：

### 登录节点信息
|Hostname|IP|OS|
|------|----------|--------|
|login1|10.1.0.101|CentOS 7|
|login2|10.1.0.102|Rocky 9|
|login9|10.1.0.109|Rocky 9|

### 账号密码
1. 账号：集群用户名
2. 密码：集群密码

### Web
5. Web支持图形界面xfwd、命令行界面ssh，文件传输sftp

### SSH
2. Username: 输入集群用户名 或 集群用户名/登录节点IP/self（如用户名是<username>需要登录到login9，则输入<username>/10.1.0.109/self）

### SFTP 
2. Username: 输入集群用户名/登录节点IP/self（如用户名是<username>需要登录到login9，则输入<username>/10.1.0.109/self）

### 注意事项

3. 必须通过作业调度系统进行计算，不得在登录节点或计算节点直接运行计算程序，否则会被杀掉进程
4. 登录后用 passwd 命令更改密码，密码至少8个字符，包含小写字母、大写字母、数字和特殊字符中至少三种，不能是常见密码


## VPN

### 南京大学 VPN
1. 使用南京大学统一身份认证账号、密码登录[南京大学VPN](https://vpn.nju.edu.cn)
2. 南京大学VPN使用方法参考[信息化建设管理服务中心网站](https://itsc.nju.edu.cn/21601/list.htm)

### eScience中心 VPN
1. 从[官网(中文)](https://www.hillstonenet.com.cn/support-and-training/hillstone-secure-connect/) [官网(English)](https://www.hillstonenet.com/more/services/product-downloads/)下载安装对应的 VPN 客户端
2. 服务器：cm.<username>.cn
3. 端口：4433
4. 用户名、密码：超算的用户名密码，注意密码不含动态口令
5. 登录后仅能通过此VPN访问eScience中心的服务

## 数据传输
1. SFTP：使用SFTP客户端浏览并传输文件，适合大文件和整个目录的上下传。
2. ZMODEM：在登录节点上可用rz上传文件、sz下载文件，适合单个小文件的上下传。
3. 云盘：通过云盘自动同步、传输和备份文件。
4. 传输节点：专门的数据传输节点。
5. 数据拷贝：直接使用移动硬盘拷贝大量数据文件。

### 传输节点

独立的数据传输节点，账号与集群用户名一致，密码独立且无动态口令。

#### [数据存储-归档存储](https://doc.nju.edu.cn/books/357a6/page/44f68)

集群借用了stor.nju.edu.cn用于数据传输节点，故请务必**首先阅读**上述**数据存储-归档存储**中[FTP](https://doc.nju.edu.cn/books/357a6/page/44f68#bkmrk-ftp)和[SFTP](https://doc.nju.edu.cn/books/357a6/page/44f68#bkmrk-sftp)内容，并**合并覆盖以下**内容：

- 账号：集群用户名
- 密码：~/stor_passwd文件中

##### 开通方式

传输节点账号采用自助开通方式，进行如下操作后需要等待大约1小时生效

- 开通账号：`touch ~/stor_passwd`
- 查看密码：`cat ~/stor_passwd`
- 重置密码：`> ~/stor_passwd`
- 禁用账号：`rm ~/stor_passwd`

为安全起见，查看完密码后建议将文件内容改为非空占位文本（如 `echo ok > ~/stor_passwd`），避免明文长期保留。
  
### 数据拷贝

为方便大数据传输，中心提供数据拷贝服务，用户需提供大容量移动存储设备，文件系统格式为 ext4/XFS/exFAT ，并告知需要拷贝数据的目录（绝对路径），保证剩余空间足够。

- 推荐使用文件系统为 ext4/XFS ，Linux可以直接读写，Windows的WSL(Windows Subsystem for Linux)也可以直接读写。
- exFAT 可以方便的同时被 Windows 和 Linux 读写，但是某些情况下无法拷贝文件，且对于机械硬盘不适于长期使用。

<!--
### 存储配额
并行文件系统fsa等有存储空间配额，存储空间收费按照配额收费。

可以使用命令 `myquota` 查看配额，用户和组配额同时生效。用户或组占用存储空间超过配额但是<10%的，可以在7天宽限期内降低至配额以内，超期将无法写入任何数据；超过配额>10%，立刻无法写入任何数据，需要降低至配额以内才能写入数据。

## 登录方式access（已废弃）

### 首次登录

1. 因为需要进行两步认证绑定，故首次登录必须使用Web浏览器
2. 在计算机上使用Web浏览器访问 [https://access.nju.edu.cn](https://access.nju.edu.cn/)
3. 账号、密码：输入用户名、密码登录
4. 手机安装支持 [RFC 6238](https://datatracker.ietf.org/doc/html/rfc6238) 的 TOTP 客户端APP，扫描二维码（也可手动输入），然后点击“完成绑定”
5. 两步认证密码：输入手机 APP 中的动态口令登录
6. 下载安装或升级客户端插件

### Web登录（图形和命令行界面）

1. 使用Web浏览器访问 [https://access.nju.edu.cn](https://access.nju.edu.cn/)
2. 账号、密码：输入用户名、密码登录
3. 两步认证密码：输入手机 APP 中的动态口令登录
4. 下载安装或升级客户端插件，需要 [Java](https://java.com/download/) 运行环境
5. Web 支持图形界面 xfce4/vnc、命令行界面 ssh ，Web 不支持 SFTP ！
6. 图形界面修改分辨率：Applications - Settings - Display - Resolution

### SSH登录（命令行界面）

1. 中国教育网地址：access.nju.edu.cn，端口：22（默认）<br>中国移动地址：njucm.<username>.com，端口：5022
2. Username: 输入用户名
3. Password: 输入密码+空格+动态口令（如密码是`<example-password>`，手机APP动态口令是`123456`，则输入`<example-password> 123456`）
4. SSH 不支持图形（X11-Forwarding）
5. SSH 客户端：[Putty](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)、[Xshell](https://www.xshell.com/free-for-home-school/)、[Bitvise SSH Client](https://www.bitvise.com/ssh-client-download)、[iTerm2](https://iterm2.com/downloads.html)、[Termius](https://www.termius.com/download)……

### SFTP登录（文件传输）

1. 中国教育网地址：access.nju.edu.cn，端口：22（默认）<br>中国移动地址：njucm.<username>.com，端口：5022
2. Username: 输入用户名
3. Password: 输入密码+空格+动态口令（如密码是`<example-password>`，手机APP动态口令是`123456`，则输入`<example-password> 123456`）
4. SFTP 路径有特殊前缀（ /hpc_login1 sftp (10.1.0.101)/self ）。如家目录为`/<home-root>/<username>`，则SFTP路径为`/hpc_login1 sftp (10.1.0.101)/self/<home-root>/<username>`
5. SFTP 客户端：[Xftp](https://www.xshell.com/free-for-home-school/)(Windows 推荐)、[WinSCP](https://winscp.net/download.php)、[Bitvise SSH Client](https://www.bitvise.com/ssh-client-download)、[Termius](https://www.termius.com/download)(macOS 推荐)、[FileZilla](https://filezilla-project.org/download.php?type=client)、[Cyberduck](https://cyberduck.io/download/)、[LFTP](http://lftp.yar.ru/get.html)
   - Xftp 如长时间连接后传输文件出错，可在“会话 (Sessions)”“属性 (Properties)”的“选项 (Options)”中勾选“仅使用主连接 (Use main connection only)”
	- WinSCP 不可使用“后台传输 (Background Transfers)”
	- FileZilla 长时间空闲不操作连接会中断，须在“站点 (Site)”的“传输设置 (Transfer Settings)”中勾选“限制并发连接数 (Limit number of simultaneous connections)”并设置“最大连接数 (Maximum number of connections)”为1
    Cyberduck 须在“新建书签 (New Bookmark)”中设置“传输文件 (Transfer Files)”为“使用浏览器连接 (Use browser connection)”，传输大文件会出错。
6. SFTP 挂载盘：[SFTP Drive](https://www.nsoftware.com/sftp/drive/download.aspx)、[SSHFS](https://github.com/libfuse/sshfs)
	- SFTP Drive 须设置“Authentication Type”为“Keyboard Interactive”

### 登录注意事项

1. 连续输错5次密码（含动态口令），账号锁定5分钟
2. 只有最新绑定的动态口令有效，为避免混淆建议在绑定前先删除老的条目，动态口令仅用于登录 access.nju.edu.cn
3. Shell命令行提示符（环境变量PS1）最后必须以`$ `结尾（$+空格）
4. 登录后用 passwd 命令更改密码，密码至少8个字符，包含小写字母、大写字母、数字和特殊字符中至少三种，不能是常见密码
5. 必须通过作业调度系统进行计算，不得在登录节点或计算节点直接运行计算程序，否则会被杀掉进程
-->