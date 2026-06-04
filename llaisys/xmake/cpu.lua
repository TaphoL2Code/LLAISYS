-- SIMD / OpenMP / OpenBLAS options
option("use-simd")
    set_default(true)
    set_showmenu(true)
    set_description("Enable SIMD (AVX2) optimizations for CPU operators")
option_end()

option("use-openmp")
    set_default(true)
    set_showmenu(true)
    set_description("Enable OpenMP multi-threading for CPU operators")
option_end()

option("use-openblas")
    set_default(false)
    set_showmenu(true)
    set_description("Link OpenBLAS for GEMM acceleration")
option_end()

target("llaisys-device-cpu")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    if has_config("use-simd") then
        add_defines("USE_SIMD")
        if is_plat("windows") then
            add_cxflags("/arch:AVX2")
        else
            add_cxflags("-mavx2", "-mfma")
        end
    end

    if has_config("use-openmp") then
        add_defines("USE_OPENMP")
        if is_plat("windows") then
            add_cxflags("/openmp")
        else
            add_cxflags("-fopenmp")
        end
    end

    add_files("../src/device/cpu/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    if has_config("use-simd") then
        add_defines("USE_SIMD")
        if is_plat("windows") then
            add_cxflags("/arch:AVX2")
        else
            add_cxflags("-mavx2", "-mfma")
        end
    end

    if has_config("use-openmp") then
        add_defines("USE_OPENMP")
        if is_plat("windows") then
            add_cxflags("/openmp")
        else
            add_cxflags("-fopenmp")
        end
    end

    if has_config("use-openblas") then
        add_defines("USE_OPENBLAS")
        add_packages("openblas")
    end

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()

