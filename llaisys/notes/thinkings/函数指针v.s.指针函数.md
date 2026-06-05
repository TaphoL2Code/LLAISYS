# 函数指针与指针函数详解

## 一、核心区别速览

| 维度 | 函数指针 | 指针函数 |
|------|---------|----------|
| **本质** | 指针，指向函数 | 函数，返回指针 |
| **语法特征** | `(*ptr)` 在类型名旁 | `*` 在函数名旁 |
| **内存** | 存储函数地址（4/8字节） | 函数体本身，返回地址值 |
| **用途** | 回调、策略模式、动态分发 | 返回动态分配的内存/对象 |

---

## 二、语法辨析

### 2.1 函数指针（Function Pointer）

**本质是一个指针变量，存储的是函数的入口地址。**

```c
// 基本语法：返回类型 (*指针变量名)(参数类型列表)

int (*func_ptr)(int, int);          // 指向返回 int、接受两个 int 参数的函数
void (*callback)(const char *);     // 指向返回 void、接受 const char* 的函数
```

**阅读规则**：从内向外读
- `func_ptr` 是一个指针 →
- 指向一个函数（`(int, int)`）→
- 该函数返回 `int`

### 2.2 指针函数（Pointer Function）

**本质是一个函数，其返回值是一个指针。**

```c
// 基本语法：返回类型* 函数名(参数类型列表)

int *get_array(int size);           // 返回 int* 的函数
char *strdup(const char *src);      // 返回 char* 的函数
void *malloc(size_t size);          // 返回 void* 的函数
```

**阅读规则**：从左向右读
- `get_array` 是一个函数 →
- 接受 `int` 参数 →
- 返回 `int*`（指向 int 的指针）

---

## 三、LLAISYS 项目中的实际应用

### 3.1 函数指针 — Runtime API 结构体

LLAISYS 中最典型的函数指针应用在 [runtime.h](file:///c:/Code/LLAISYS/llaisys/include/llaisys/runtime.h) 中：

```c
// 第一步：使用 typedef 定义函数指针类型
typedef int    (*get_device_count_api)();
typedef void   (*set_device_api)(int);
typedef void   (*device_synchronize_api)();
typedef void * (*malloc_device_api)(size_t);
typedef void   (*free_device_api)(void *);
typedef void   (*memcpy_sync_api)(void *, const void *, size_t, llaisysMemcpyKind_t);

// 第二步：将函数指针聚合为结构体（虚表 / vtable 模式）
struct LlaisysRuntimeAPI {
    get_device_count_api    get_device_count;
    set_device_api          set_device;
    device_synchronize_api  device_synchronize;
    malloc_device_api       malloc_device;
    free_device_api         free_device;
    memcpy_sync_api         memcpy_sync;
    // ...
};
```

**设计意图**：这是一个手动的虚函数表（vtable），用于实现**运行时多态**。CPU 和 NVIDIA GPU 各自提供一套实现：

```c
// CPU 实现：将函数地址填入结构体
static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,      // 函数名本身即地址
    &setDevice,
    &deviceSynchronize,
    &mallocDevice,
    &freeDevice,
    &memcpySync,          // CPU 版 memcpySync 内部调用 std::memcpy
    // ...
};

// 调用方式：通过结构体成员间接调用
const LlaisysRuntimeAPI *api = llaisysGetRuntimeAPI(LLAISYS_DEVICE_CPU);
api->memcpy_sync(dst, src, size, LLAISYS_MEMCPY_H2D);  // 实际调用 CPU 版本
```

### 3.2 指针函数 — 内存分配函数

同一文件中，内存分配函数本身就是指针函数：

```c
// 这些是典型的指针函数：返回 void* 指针
typedef void *(*malloc_device_api)(size_t);   // 返回指向设备内存的指针
typedef void *(*malloc_host_api)(size_t);     // 返回指向主机内存的指针
```

实际的 CPU 实现（[cpu_runtime_api.cpp](file:///c:/Code/LLAISYS/llaisys/src/device/cpu/cpu_runtime_api.cpp)）：

```c
void *mallocDevice(size_t size) {
    return std::malloc(size);   // 返回 void* 指针 — 这就是指针函数
}

void freeDevice(void *ptr) {
    std::free(ptr);             // 接收指针参数，返回 void
}
```

### 3.3 两者结合 — 函数指针指向指针函数

在 LLAISYS 中，函数指针和指针函数经常组合使用：

```c
// malloc_device 是一个函数指针，指向一个指针函数
malloc_device_api malloc_device;  // 函数指针类型
//                                 ↓
// 它指向的函数签名为: void* func(size_t)  ← 这是一个指针函数

// 调用时：
void *ptr = api->malloc_device(1024);  // 通过函数指针调用指针函数
```

---

## 四、typedef 简化对比

```c
// ========== 函数指针的 typedef ==========
// 语法：typedef 返回类型 (*新类型名)(参数列表);

typedef int (*MathOp)(int, int);    // MathOp 是一个函数指针类型
MathOp add = &my_add;              // 声明函数指针变量
int result = add(3, 5);            // 通过函数指针调用


// ========== 指针函数的 typedef ==========
// 语法：typedef 返回类型* (*新类型名)(参数列表);
//       注意：此时 typedef 的是函数指针类型，而非指针函数类型本身

// C 语言中无法直接 typedef 一个"指针函数类型"
// 但可以这样用：
typedef void* (*Allocator)(size_t);  // 这实际上是函数指针的 typedef
Allocator my_alloc = &malloc;        // 指向返回 void* 的函数


// ========== 正确做法：先 typedef 函数签名，再取指针 ==========
typedef void* AllocFunc(size_t);     // AllocFunc 是函数类型
AllocFunc *my_alloc = &malloc;       // my_alloc 是函数指针
```

---

## 五、从简单到复杂的演变

```
第一层：普通函数
    int add(int a, int b) { return a + b; }

第二层：指针函数（返回指针的函数）
    int* create_array(int n) { return malloc(n * sizeof(int)); }

第三层：函数指针（指向函数的指针）
    int (*op)(int, int) = &add;

第四层：函数指针指向指针函数（最复杂）
    void* (*alloc)(size_t) = &malloc;  // alloc 指向返回 void* 的函数
    void* ptr = alloc(100);           // 调用 malloc
```

---

## 六、常见误区与陷阱

### 误区 1：语法混淆

```c
int  *p(int);    // 指针函数：函数 p 返回 int*
int (*q)(int);   // 函数指针：q 是指向函数的指针

// 区别在于括号的位置！
```

### 误区 2：函数指针赋值时 & 可省略

```c
int (*fp)(int, int) = &add;   // 正确，显式取地址
int (*fp)(int, int) = add;    // 也正确，函数名自动退化为指针

fp(3, 5);                     // 调用方式 1
(*fp)(3, 5);                  // 调用方式 2，等价
```

### 误区 3：返回局部变量的指针

```c
// 错误！返回了栈上局部变量的地址
int* bad_func() {
    int x = 42;
    return &x;   // 函数返回后 x 被销毁，指针悬空
}

// 正确做法：返回堆内存或静态变量
int* good_func() {
    int* p = malloc(sizeof(int));
    *p = 42;
    return p;    // 调用者负责 free
}
```

### 误区 4：函数指针数组声明

```c
// 最外层是数组，元素是函数指针
int (*handlers[5])(int, int);

// 等价于：
typedef int (*Handler)(int, int);
Handler handlers[5];
```

---

## 七、总结

| 记忆口诀 | 说明 |
|----------|------|
| **函数指针** | "指针"是主角，它是一个**变量**，存的是函数地址 |
| **指针函数** | "函数"是主角，它是一个**函数**，返回值是指针 |

```
函数指针：  int (*p)(int)     → p 是变量，存地址
指针函数：  int  *f(int)      → f 是函数，返回指针
```

在 LLAISYS 项目中，`LlaisysRuntimeAPI` 结构体是函数指针的集大成应用——通过将函数指针组织为结构体成员，实现了 C 语言层面的**策略模式**和**依赖注入**，使得同一套 API 可以在 CPU 和 GPU 之间无缝切换。