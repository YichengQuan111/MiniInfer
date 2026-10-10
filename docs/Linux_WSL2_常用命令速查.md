# Linux / WSL2 常用命令速查

适用于 Ubuntu 24.04 和 MiniInfer 开发。除最后一节外，命令均在 Ubuntu 终端中执行。示例中的文件名、包名和进程编号需按实际情况替换。

## 1. 目录与路径

| 命令 | 用途 |
|---|---|
| `pwd` | 显示当前目录 |
| `ls` | 列出当前目录内容 |
| `ls -lah` | 显示隐藏文件、权限及易读的文件大小 |
| `cd ~/projects/MiniInfer` | 进入项目目录 |
| `cd ..` | 回到上一级目录 |
| `cd ~` | 回到当前 Linux 用户的主目录 |
| `cd -` | 回到上一次所在的目录 |
| `mkdir -p ~/projects/MiniInfer` | 创建目录，必要时同时创建父目录 |

路径提示：`~` 表示当前用户的主目录；`.` 表示当前目录；`..` 表示父目录。Linux 文件名区分大小写。

Windows 的 C、D 盘通常分别位于 `/mnt/c`、`/mnt/d`。若 Ubuntu 虚拟磁盘装在 D 盘，`~/projects/MiniInfer` 内的文件也存储在 D 盘的虚拟磁盘中。

## 2. 文件操作

| 命令 | 用途 |
|---|---|
| `touch example.txt` | 新建空文件；文件已存在时更新其时间戳 |
| `cp example.txt example.bak` | 复制文件 |
| `cp -a source_dir backup_dir` | 复制目录并保留属性；目标不存在时创建目标副本，存在时通常复制到其内部 |
| `mv old.txt new.txt` | 重命名文件 |
| `mv example.txt target_dir/` | 移动文件到已有目录 |
| `rm -i example.txt` | 删除文件，先询问确认 |
| `rm -ri example_dir` | 递归删除目录，逐项询问确认 |
| `rmdir empty_dir` | 删除空目录 |

`cp`、`mv` 可能覆盖同名文件；需要确认时加 `-i`。`rm` 删除通常不进入回收站。

## 3. 查看、编辑与查找

| 命令 | 用途 |
|---|---|
| `cat example.txt` | 查看短文件的完整内容 |
| `less example.txt` | 分页查看文件；按 `q` 退出 |
| `head -n 20 example.txt` | 查看前 20 行 |
| `tail -n 50 example.log` | 查看最后 50 行 |
| `tail -f example.log` | 持续查看新增日志；按 `Ctrl+C` 结束查看 |
| `nano example.txt` | 编辑文件；`Ctrl+O` 保存，回车确认文件名，`Ctrl+X` 退出 |
| `find . -type f -name '*.cpp'` | 在当前目录及子目录中查找 C++ 文件 |
| `grep -n '关键词' example.txt` | 查找文件中的文本并显示行号 |
| `rg -n 'Tensor' include src` | 在指定目录快速查找文本；需要安装 ripgrep |
| `rg --files -g '*.cpp'` | 列出匹配的文件；需要安装 ripgrep |

可选安装：`sudo apt install ripgrep nano`。查找大文件时，优先按关键词筛选，避免输出整个文件。

## 4. 软件安装与权限

| 命令 | 用途 |
|---|---|
| `sudo apt update` | 更新可用软件包列表 |
| `sudo apt install cmake` | 安装软件包；此处以 CMake 为例 |
| `sudo apt upgrade` | 升级已安装的软件包，先查看待升级内容 |
| `apt search cmake` | 搜索软件包 |
| `whoami` | 显示当前用户名 |
| `id` | 显示用户及用户组信息 |
| `chmod +x script.sh` | 给脚本增加执行权限 |
| `./script.sh` | 执行当前目录中的脚本 |

`sudo` 用于以管理员权限运行指定命令。输入密码时不会显示字符或星号。项目文件的日常操作通常使用普通用户即可。

## 5. 系统与进程

| 命令 | 用途 |
|---|---|
| `uname -a` | 查看内核信息 |
| `uname -m` | 查看架构；本项目目标为 `x86_64` |
| `df -h` | 查看文件系统的已用和可用空间 |
| `du -sh .` | 统计当前目录占用的磁盘空间 |
| `free -h` | 查看当前环境的内存与交换空间 |
| `nproc` | 查看可用逻辑处理器数量 |
| `ps -ef` | 列出进程 |
| `top` | 动态查看 CPU、内存及进程；按 `q` 退出 |
| `kill PID` | 向进程发送终止信号；把 PID 替换为确认过的数字进程编号 |

WSL 中 `df -h` 显示的是虚拟文件系统空间；还应在 Windows 中确认存放虚拟磁盘的盘符有足够实际剩余空间。

## 6. MiniInfer 的开发命令

### Python 参考模型环境

按顺序运行以下命令：

```bash
cd ~/projects/MiniInfer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scikit-learn
```

`.venv` 只需创建一次；每次新开终端时按需激活。使用 `python -m pip list` 查看环境中的包，使用 `deactivate` 退出虚拟环境。

### C++ / CMake

当前项目只有规划文档。下列构建命令用于后续已建立 `CMakeLists.txt` 的工程，并要求已安装对应工具；CTest 还要求工程注册了测试。

| 命令 | 用途 |
|---|---|
| `g++ --version` | 查看 GCC C++ 编译器版本 |
| `clang++ --version` | 查看 Clang C++ 编译器版本 |
| `cmake --version` | 查看 CMake 版本 |
| `cmake -S . -B build/debug -G Ninja -DCMAKE_BUILD_TYPE=Debug` | 配置 Debug 构建目录 |
| `cmake --build build/debug -j 4` | 编译，最多使用 4 个并行任务 |
| `ctest --test-dir build/debug --output-on-failure` | 运行测试，失败时显示输出；应确认实际执行了测试 |
| `gdb ./app` | 调试已有的 app 可执行文件 |

### Git

| 命令 | 用途 |
|---|---|
| `git init` | 在尚未初始化的项目目录中创建本地仓库 |
| `git status` | 查看改动与暂存状态 |
| `git diff` | 查看尚未暂存的修改 |
| `git add example.cpp` | 暂存指定文件的修改 |
| `git diff --staged` | 查看已经暂存的修改 |
| `git commit -m '描述本次改动'` | 提交已经暂存的修改，需要先配置 Git 身份 |
| `git log --oneline -n 5` | 查看最近 5 次提交 |

将构建目录与 Python 虚拟环境加入项目的 `.gitignore`，例如忽略 `build/` 和 `.venv/`。

## 7. 帮助、快捷键与命令连接

| 操作 | 用途 |
|---|---|
| `ls --help` | 查看命令的简要用法 |
| `man ls` | 查看命令手册；按 `q` 退出，需要已安装对应手册 |
| `history` | 查看命令历史 |
| `clear` 或 `Ctrl+L` | 清理终端显示 |
| `Tab` | 自动补全命令、文件名或路径 |
| 上 / 下方向键 | 浏览之前输入的命令 |
| `Ctrl+C` | 请求中断前台命令 |
| `exit` | 退出当前 shell；WSL 中通常返回 Windows 终端 |

在 Windows Terminal 中，复制通常用 `Ctrl+Shift+C`，粘贴用 `Ctrl+Shift+V`。旧版 Windows 控制台可选中文字后按回车复制、用右键粘贴；实际行为取决于终端设置。

`命令1 && 命令2`：命令1 成功后才运行命令2。

`命令1 | 命令2`：把前一个命令的标准输出交给后一个命令处理。

`命令 > result.txt`：把标准输出写入文件，会覆盖已有内容。

`命令 >> result.txt`：把标准输出追加到文件末尾。

## 8. Windows 与 WSL 管理命令

| 在 Windows PowerShell 中执行 | 用途 |
|---|---|
| `wsl --list --verbose` | 查看发行版及其 WSL 版本 |
| `wsl --set-default Ubuntu-24.04` | 设置默认发行版 |
| `wsl -d Ubuntu-24.04 --cd ~` | 打开 Ubuntu 并进入主目录 |
| `wsl --version` | 查看 WSL 组件版本 |

在 Ubuntu 终端里执行上述管理操作时，将 `wsl` 换成 `wsl.exe`。例如：

```bash
wsl.exe --list --verbose
wsl.exe --set-default Ubuntu-24.04
```

`explorer.exe .` 可在 Windows 文件资源管理器中打开当前 Linux 目录。

安装并配置好 VS Code 的 WSL 扩展后，在项目目录执行 `code .` 可打开 WSL 开发窗口。
