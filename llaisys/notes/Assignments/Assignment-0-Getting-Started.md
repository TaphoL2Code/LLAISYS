# Assignment #0: Getting Started — 事件清单

## 主要修改文件
**无需修改代码**，只需完成环境搭建和验证。

## 需更改的配置
**无需更改配置**。

## 建议阅读层次

> 参考 [遍历式阅读顺序](../thinking/遍历式阅读顺序.md)

| 层次 | 内容 | 关注点 |
|:--:|------|------|
| **第 0 层** | [include/](file:///c:/Code/LLAISYS/llaisys/include/) — 公共头文件 | 认识 `llaisys.h` 中所有枚举、`ops.h`/`tensor.h`/`runtime.h` 中的 API 签名 |
| **第 11 层** | [xmake.lua](file:///c:/Code/LLAISYS/llaisys/xmake.lua) + [xmake/cpu.lua](file:///c:/Code/LLAISYS/llaisys/xmake/cpu.lua) | 理解 9 个 target 的依赖关系和编译选项 |
| **第 12 层** | [test/](file:///c:/Code/LLAISYS/llaisys/test/) — 测试 | 理解 `test_utils.py` 的 `check_equal()` 比对逻辑、`test_infer.py` 的推理验证流程 |

---

## 任务清单

### 任务 0.1：安装必要开发环境

- [x] **安装 Xmake 构建工具**
  - 去 [xmake.io](https://xmake.io/) 按照指南安装
  - 验证安装成功：`xmake --version`

验证：

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-0-Getting-Started\Snipaste_2026-06-01_03-26-06.png)

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-0-Getting-Started\Snipaste_2026-06-01_03-28-32.png)

- [x] **安装 C++ 编译器**
  - Windows：安装 Visual Studio 2022（含 MSVC）或 Clang
  - Linux：安装 GCC 或 Clang（`sudo apt install build-essential`）
  - 验证安装成功：`gcc --version` 或 `clang --version`

验证：

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-0-Getting-Started\Snipaste_2026-06-01_03-29-41.png)

- [x] **安装 Python ≥ 3.9 及依赖库**

  - 确保 Python 版本 ≥ 3.9：`python --version`

  - 预先安装 PyTorch、Transformers、Accelerate：

    ```bash
    pip install torch transformers accelerate safetensors
    ```

- [ ] **（可选）安装 Clang-Format-16**

  - 用于 C++ 代码格式化
  - `pip install clang-format` 或根据系统安装

验证：

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-0-Getting-Started\Snipaste_2026-06-01_03-30-24.png)

### 任务 0.2：Fork 并构建 LLAISYS

- [x] **Fork 仓库**
  - 在 GitHub 上 Fork LLAISYS 仓库到自己账号

- [x] **克隆 Fork 后的仓库到本地**
  - `git clone <你的Fork地址>`
  - `cd llaisys`

- [x] **编译 C++ 代码**
  
  - ```bash
    xmake
    ```
  - 确认编译无错误

验证：

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-0-Getting-Started\Snipaste_2026-06-01_03-39-18.png)

- [x] **安装 LLAISYS 共享库**

  - ```bash
    xmake install
    ```
  - 确认 `.dll`（Windows）或 `.so`（Linux）被复制到 `python/llaisys/libllaisys/`

验证：

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-0-Getting-Started\Snipaste_2026-06-01_03-39-53.png)

- [x] **安装 LLAISYS Python 包**
  - ```bash
    pip install ./python/
    ```
  - 验证：`python -c "import llaisys; print(llaisys.Tensor)"` 无报错

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-0-Getting-Started\Snipaste_2026-06-01_03-42-03.png)

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-0-Getting-Started\Snipaste_2026-06-01_03-42-24.png)

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-0-Getting-Started\Snipaste_2026-06-01_03-42-36.png)

- [x] **启用 GitHub Actions 自动测试**
  - 进入自己仓库的 Settings → Actions → 确保启用
  - 后续每次 push 都会自动运行 `.github/workflows/build.yaml` 中的测试

验证：

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-0-Getting-Started\Snipaste_2026-06-01_03-46-57.png)

### 任务 0.3：首次运行 LLAISYS

- [x] **运行 CPU 运行时测试**
  - ```bash
    python test/test_runtime.py --device cpu
    ```
  - 预期看到绿色 `Test passed!`

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-0-Getting-Started\Snipaste_2026-06-01_03-47-42.png)

### 任务 0.4：下载测试模型

- [x] **下载 DeepSeek-R1-Distill-Qwen-1.5B 模型**
  - 从 HuggingFace 下载：https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  - 或使用 Python 自动下载（首次运行 `test_infer.py` 时会自动下载）

- [x] **使用 PyTorch 运行推理测试，确认模型可用**
  - ```bash
    python test/test_infer.py --model <模型路径>
    ```
  - 预期能看到 PyTorch 正常加载模型并生成文本
  - 此时 LLAISYS 部分会因尚未实现而失败，这是正常的

验证：

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-0-Getting-Started\Snipaste_2026-06-01_03-50-54.png)

---

## 验证标准
- [x] `xmake` 编译成功
- [x] `xmake install` 安装成功
- [x] `pip install ./python/` 安装成功
- [x] `python test/test_runtime.py --device cpu` 通过
- [x] `python test/test_infer.py --model <模型路径>` 中 PyTorch 推理部分正常运行
- [x] GitHub Actions 自动构建通过（push 后查看）