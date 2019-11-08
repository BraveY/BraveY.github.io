---
title: apt-get install失败
date: 2019-03-10 16:23:20
categories: Linux
tags: 
- Linux 
- 运维
copyright: true
---

## apt-get install失败

<!--more-->

### 第一阶段

1. 使用perf 报错 内核无法找到perf

   ```
   root@hw103:/home/yky/redis-5.0.3# perf 
   WARNING: perf not found for kernel 4.15.0-45
   
     You may need to install the following packages for this specific kernel:
       linux-tools-4.15.0-45-generic
       linux-cloud-tools-4.15.0-45-generic
   
     You may also want to install one of the following packages to keep up to date:
       linux-tools-generic
       linux-cloud-tools-generic
   ```

   

2. 安装此内核的通用工具时错误

   ```
   root@hw103:/home/yky/redis-5.0.3# apt-get install  linux-tools-4.15.0-45-generic
   Reading package lists... Done
   Building dependency tree       
   Reading state information... Done
   You might want to run 'apt-get -f install' to correct these:
   The following packages have unmet dependencies:
    console-setup : Depends: keyboard-configuration (= 1.178ubuntu2.7) but 1.108ubuntu15.3 is to be installed
    console-setup-linux : Depends: keyboard-configuration (= 1.178ubuntu2.7) but 1.108ubuntu15.3 is to be installed
                          Breaks: keyboard-configuration (< 1.138) but 1.108ubuntu15.3 is to be installed
    linux-tools-4.15.0-45-generic : Depends: linux-tools-4.15.0-45 but it is not going to be installed
   E: Unmet dependencies. Try 'apt-get -f install' with no packages (or specify a solution).
   ```

3. 使用apt-get -f install 时报错

   ```
   update-rc.d: error: insserv rejected the script header
   dpkg: error processing archive /var/cache/apt/archives/keyboard-configuration_1.178ubuntu2.7_all.deb (--unpack):
    subprocess new pre-installation script returned error exit status 1
   dpkg-query: warning: files list file for package 'keyboard-configuration' missing; assuming package has no files currently installed
   dpkg-query: warning: files list file for package 'keyboard-configuration' missing; assuming package has no files currently installed
   dpkg-query: warning: files list file for package 'keyboard-configuration' missing; assuming package has no files currently installed
   Errors were encountered while processing:
    /var/cache/apt/archives/keyboard-configuration_1.178ubuntu2.7_all.deb
   E: Sub-process /usr/bin/dpkg returned an error code (1)
   ```

   问题综述：

   1. `apt-get install lib`时报错 Unmet dependencies
   2. `apt-get install -f ` 时报错Sub-process /usr/bin/dpkg returned an error code (1)


4. 第一阶段解决办法

   在/var/lib/dpkg/目录下有个info文件 ，然后文件中没有keyboard-configuration的相关文件但是有info的备份info_backup  ，这里面有相关的文件，于是将keyboard-configuration的所有相关文件都拷贝到了/var/lib/dpkg/info 中。

   在info_backup目录下执行如下命令拷贝

   `cp keyboard-configuration.* ../info`

   随后再次执行安装内核通用工具 报错为第二阶段

### 第二阶段

1. 安装此内核的通用工具时时报错：

   ``` 
   insserv: Starting redis depends on plymouth and therefore on system facility `$all' which can not be true!
   insserv: exiting now without changing boot order!
   update-rc.d: error: insserv rejected the script header
   dpkg: error processing package avahi-daemon (--configure):
    subprocess installed post-installation script returned error exit status 1
   No apport report written because MaxReports is reached already
                                                                 No apport report written because MaxReports is reached already
                                                                                                                               dpkg: dependency problems prevent configuration o
   f avahi-utils: avahi-utils depends on avahi-daemon; however:
     Package avahi-daemon is not configured yet.
   
   dpkg: error processing package avahi-utils (--configure):
    dependency problems - leaving unconfigured
   Setting up unattended-upgrades (1.1ubuntu1.18.04.9) ...
   dpkg: error processing package unattended-upgrades (--configure):
    subprocess installed post-installation script returned error exit status 10
   No apport report written because MaxReports is reached already
                                                                 Setting up linux-tools-4.15.0-45 (4.15.0-45.48) ...
   Setting up linux-tools-4.15.0-45-generic (4.15.0-45.48) ...
   Processing triggers for initramfs-tools (0.122ubuntu8.14) ...
   Errors were encountered while processing:
    udev
    snapd
    ubuntu-core-launcher
    kmod
    ubuntu-drivers-common
    whoopsie
    openssh-server
    ssh
    avahi-daemon
    avahi-utils
    unattended-upgrades
   E: Sub-process /usr/bin/dpkg returned an error code (1)
   ```

2. 解决办法：/var/lib/dpkg/info 目录下将上述出现问题的模块的postinst文件重命名。

   在/var/lib/dpkg/info 下写了个脚本

   solution.sh

   ```
   #!/bin/bash
   for pack in $(cat module.txt)
   do 
       mv "$pack".postinst "$pack".postinst.bak
   done
   ```

   其中module.txt的内容为

   ```
    udev
    snapd
    ubuntu-core-launcher
    kmod
    ubuntu-drivers-common
    whoopsie
    openssh-server
    ssh
    avahi-daemon
    avahi-utils
    unattended-upgrades
   ```

3. 执行脚本后 使用`sudo apt-get upgrade` 进行更新

4. 参考：

   1. https://www.codelast.com/%E5%8E%9F%E5%88%9B-%E8%A7%A3%E5%86%B3ubuntu-%E6%97%A0%E6%B3%95%E7%94%A8-apt-get-install-%E5%AE%89%E8%A3%85%E4%BB%BB%E4%BD%95%E8%BD%AF%E4%BB%B6dpkg-error-processing-package-xxx%E7%9A%84%E9%97%AE/
   2. https://askubuntu.com/questions/949760/dpkg-warning-files-list-file-for-package-missing