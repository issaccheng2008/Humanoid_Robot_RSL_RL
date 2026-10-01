#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"
#include "deterministic.h"

// Map wp.breakpoint() to a device brkpt at the call site so cuda-gdb attributes the stop to the generated .cu line
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()

// CUDA Thread Block Cluster shape declaration. Expands to __cluster_dims__
// only on devices that support clusters (compute capability 9.0+); otherwise
// expands to nothing so the same source compiles cleanly for any target arch.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define WP_CLUSTER_DIMS(x, y, z) __cluster_dims__(x, y, z)
#else
#define WP_CLUSTER_DIMS(x, y, z)
#endif



extern "C" __global__ void update_outdated_envs_kernel_937ac92b_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<bool> var_is_outdated,
    wp::array_t<wp::float32> var_timestamp,
    wp::array_t<wp::float32> var_timestamp_last_update)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        bool* var_1;
        bool var_2;
        wp::float32* var_3;
        wp::float32 var_4;
        const bool var_5 = false;
        bool var_6;
        //---------
        // forward
        // def update_outdated_envs_kernel(                                                       <L 34>
        // env = wp.tid()                                                                         <L 46>
        var_0 = builtin_tid1d();
        // if is_outdated[env]:                                                                   <L 47>
        var_1 = wp::address(var_is_outdated, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // timestamp_last_update[env] = timestamp[env]                                        <L 48>
            var_3 = wp::address(var_timestamp, var_0);
            var_4 = wp::load(var_3);
            wp::array_store(var_timestamp_last_update, var_0, var_4);
            // is_outdated[env] = False                                                           <L 49>
            wp::array_store(var_is_outdated, var_0, var_5);
        }
        var_6 = wp::load(var_1);
    }
}



extern "C" __global__ void update_outdated_envs_kernel_937ac92b_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<bool> var_is_outdated,
    wp::array_t<wp::float32> var_timestamp,
    wp::array_t<wp::float32> var_timestamp_last_update,
    wp::array_t<bool> adj_is_outdated,
    wp::array_t<wp::float32> adj_timestamp,
    wp::array_t<wp::float32> adj_timestamp_last_update)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        bool* var_1;
        bool var_2;
        wp::float32* var_3;
        wp::float32 var_4;
        const bool var_5 = false;
        bool var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        bool adj_2 = {};
        wp::float32 adj_3 = {};
        wp::float32 adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        //---------
        // forward
        // def update_outdated_envs_kernel(                                                       <L 34>
        // env = wp.tid()                                                                         <L 46>
        var_0 = builtin_tid1d();
        // if is_outdated[env]:                                                                   <L 47>
        var_1 = wp::address(var_is_outdated, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // timestamp_last_update[env] = timestamp[env]                                        <L 48>
            var_3 = wp::address(var_timestamp, var_0);
            var_4 = wp::load(var_3);
            // wp::array_store(var_timestamp_last_update, var_0, var_4);
            // is_outdated[env] = False                                                           <L 49>
            // wp::array_store(var_is_outdated, var_0, var_5);
        }
        var_6 = wp::load(var_1);
        //---------
        // reverse
        if (var_6) {
            wp::adj_array_store(var_is_outdated, var_0, var_5, adj_is_outdated, adj_0, adj_5);
            // adj: is_outdated[env] = False                                                      <L 49>
            wp::adj_array_store(var_timestamp_last_update, var_0, var_4, adj_timestamp_last_update, adj_0, adj_3);
            wp::adj_address(var_timestamp, var_0, adj_timestamp, adj_0, adj_3);
            // adj: timestamp_last_update[env] = timestamp[env]                                   <L 48>
        }
        wp::adj_address(var_is_outdated, var_0, adj_is_outdated, adj_0, adj_1);
        // adj: if is_outdated[env]:                                                              <L 47>
        // adj: env = wp.tid()                                                                    <L 46>
        // adj: def update_outdated_envs_kernel(                                                  <L 34>
        continue;
    }
}



extern "C" __global__ void update_timestamp_kernel_71be0a73_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<bool> var_is_outdated,
    wp::array_t<wp::float32> var_timestamp,
    wp::array_t<wp::float32> var_timestamp_last_update,
    wp::float32 var_dt,
    wp::float32 var_update_period)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::float32* var_1;
        wp::float32 var_2;
        wp::float32 var_3;
        wp::float32* var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        const wp::float32 var_7 = 1e-06;
        wp::float32 var_8;
        bool var_9;
        const bool var_10 = true;
        //---------
        // forward
        // def update_timestamp_kernel(                                                           <L 10>
        // env = wp.tid()                                                                         <L 26>
        var_0 = builtin_tid1d();
        // new_timestamp = timestamp[env] + dt                                                    <L 27>
        var_1 = wp::address(var_timestamp, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::add(var_3, var_dt);
        // timestamp[env] = new_timestamp                                                         <L 28>
        wp::array_store(var_timestamp, var_0, var_2);
        // if new_timestamp - timestamp_last_update[env] + 1e-6 >= update_period:                 <L 29>
        var_4 = wp::address(var_timestamp_last_update, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::sub(var_2, var_6);
        var_8 = wp::add(var_5, var_7);
        var_9 = (var_8 >= var_update_period);
        if (var_9) {
            // is_outdated[env] = True                                                            <L 30>
            wp::array_store(var_is_outdated, var_0, var_10);
        }
    }
}



extern "C" __global__ void update_timestamp_kernel_71be0a73_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<bool> var_is_outdated,
    wp::array_t<wp::float32> var_timestamp,
    wp::array_t<wp::float32> var_timestamp_last_update,
    wp::float32 var_dt,
    wp::float32 var_update_period,
    wp::array_t<bool> adj_is_outdated,
    wp::array_t<wp::float32> adj_timestamp,
    wp::array_t<wp::float32> adj_timestamp_last_update,
    wp::float32 adj_dt,
    wp::float32 adj_update_period)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::float32* var_1;
        wp::float32 var_2;
        wp::float32 var_3;
        wp::float32* var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        const wp::float32 var_7 = 1e-06;
        wp::float32 var_8;
        bool var_9;
        const bool var_10 = true;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::float32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        bool adj_9 = {};
        bool adj_10 = {};
        //---------
        // forward
        // def update_timestamp_kernel(                                                           <L 10>
        // env = wp.tid()                                                                         <L 26>
        var_0 = builtin_tid1d();
        // new_timestamp = timestamp[env] + dt                                                    <L 27>
        var_1 = wp::address(var_timestamp, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::add(var_3, var_dt);
        // timestamp[env] = new_timestamp                                                         <L 28>
        // wp::array_store(var_timestamp, var_0, var_2);
        // if new_timestamp - timestamp_last_update[env] + 1e-6 >= update_period:                 <L 29>
        var_4 = wp::address(var_timestamp_last_update, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::sub(var_2, var_6);
        var_8 = wp::add(var_5, var_7);
        var_9 = (var_8 >= var_update_period);
        if (var_9) {
            // is_outdated[env] = True                                                            <L 30>
            // wp::array_store(var_is_outdated, var_0, var_10);
        }
        //---------
        // reverse
        if (var_9) {
            wp::adj_array_store(var_is_outdated, var_0, var_10, adj_is_outdated, adj_0, adj_10);
            // adj: is_outdated[env] = True                                                       <L 30>
        }
        wp::adj_add(var_5, var_7, adj_5, adj_7, adj_8);
        wp::adj_sub(var_2, var_6, adj_2, adj_4, adj_5);
        wp::adj_address(var_timestamp_last_update, var_0, adj_timestamp_last_update, adj_0, adj_4);
        // adj: if new_timestamp - timestamp_last_update[env] + 1e-6 >= update_period:            <L 29>
        wp::adj_array_store(var_timestamp, var_0, var_2, adj_timestamp, adj_0, adj_2);
        // adj: timestamp[env] = new_timestamp                                                    <L 28>
        wp::adj_add(var_3, var_dt, adj_1, adj_dt, adj_2);
        wp::adj_address(var_timestamp, var_0, adj_timestamp, adj_0, adj_1);
        // adj: new_timestamp = timestamp[env] + dt                                               <L 27>
        // adj: env = wp.tid()                                                                    <L 26>
        // adj: def update_timestamp_kernel(                                                      <L 10>
        continue;
    }
}



extern "C" __global__ void reset_envs_kernel_b0e3ad58_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<bool> var_reset_mask,
    wp::array_t<bool> var_is_outdated,
    wp::array_t<wp::float32> var_timestamp,
    wp::array_t<wp::float32> var_timestamp_last_update)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        bool* var_1;
        bool var_2;
        bool var_3;
        const wp::float32 var_4 = 0.0;
        const wp::float32 var_5 = 0.0;
        const bool var_6 = true;
        //---------
        // forward
        // def reset_envs_kernel(                                                                 <L 53>
        // env = wp.tid()                                                                         <L 68>
        var_0 = builtin_tid1d();
        // if not reset_mask[env]:                                                                <L 69>
        var_1 = wp::address(var_reset_mask, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::unot(var_3);
        if (var_2) {
            // return                                                                             <L 70>
            continue;
        }
        // timestamp[env] = 0.0                                                                   <L 73>
        wp::array_store(var_timestamp, var_0, var_4);
        // timestamp_last_update[env] = 0.0                                                       <L 75>
        wp::array_store(var_timestamp_last_update, var_0, var_5);
        // is_outdated[env] = True                                                                <L 77>
        wp::array_store(var_is_outdated, var_0, var_6);
    }
}



extern "C" __global__ void reset_envs_kernel_b0e3ad58_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<bool> var_reset_mask,
    wp::array_t<bool> var_is_outdated,
    wp::array_t<wp::float32> var_timestamp,
    wp::array_t<wp::float32> var_timestamp_last_update,
    wp::array_t<bool> adj_reset_mask,
    wp::array_t<bool> adj_is_outdated,
    wp::array_t<wp::float32> adj_timestamp,
    wp::array_t<wp::float32> adj_timestamp_last_update)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        bool* var_1;
        bool var_2;
        bool var_3;
        const wp::float32 var_4 = 0.0;
        const wp::float32 var_5 = 0.0;
        const bool var_6 = true;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        wp::float32 adj_4 = {};
        wp::float32 adj_5 = {};
        bool adj_6 = {};
        //---------
        // forward
        // def reset_envs_kernel(                                                                 <L 53>
        // env = wp.tid()                                                                         <L 68>
        var_0 = builtin_tid1d();
        // if not reset_mask[env]:                                                                <L 69>
        var_1 = wp::address(var_reset_mask, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::unot(var_3);
        if (var_2) {
            // return                                                                             <L 70>
            goto label0;
        }
        // timestamp[env] = 0.0                                                                   <L 73>
        // wp::array_store(var_timestamp, var_0, var_4);
        // timestamp_last_update[env] = 0.0                                                       <L 75>
        // wp::array_store(var_timestamp_last_update, var_0, var_5);
        // is_outdated[env] = True                                                                <L 77>
        // wp::array_store(var_is_outdated, var_0, var_6);
        //---------
        // reverse
        wp::adj_array_store(var_is_outdated, var_0, var_6, adj_is_outdated, adj_0, adj_6);
        // adj: is_outdated[env] = True                                                           <L 77>
        wp::adj_array_store(var_timestamp_last_update, var_0, var_5, adj_timestamp_last_update, adj_0, adj_5);
        // adj: timestamp_last_update[env] = 0.0                                                  <L 75>
        wp::adj_array_store(var_timestamp, var_0, var_4, adj_timestamp, adj_0, adj_4);
        // adj: timestamp[env] = 0.0                                                              <L 73>
        if (var_2) {
            label0:;
            // adj: return                                                                        <L 70>
        }
        wp::adj_address(var_reset_mask, var_0, adj_reset_mask, adj_0, adj_1);
        // adj: if not reset_mask[env]:                                                           <L 69>
        // adj: env = wp.tid()                                                                    <L 68>
        // adj: def reset_envs_kernel(                                                            <L 53>
        continue;
    }
}

