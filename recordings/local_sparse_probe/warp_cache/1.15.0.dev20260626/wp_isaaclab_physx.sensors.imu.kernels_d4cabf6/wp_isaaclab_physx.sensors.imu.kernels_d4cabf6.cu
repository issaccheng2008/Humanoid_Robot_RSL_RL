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



extern "C" __global__ void imu_reset_kernel_c1ffcda0_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_ang_vel_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_lin_acc_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_prev_lin_vel_w)
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
        const wp::float32 var_6 = 0.0;
        wp::vec_t<3, wp::float32> var_7;
        const wp::float32 var_8 = 0.0;
        const wp::float32 var_9 = 0.0;
        const wp::float32 var_10 = 0.0;
        wp::vec_t<3, wp::float32> var_11;
        const wp::float32 var_12 = 0.0;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        wp::vec_t<3, wp::float32> var_15;
        //---------
        // forward
        // def imu_reset_kernel(                                                                  <L 71>
        // idx = wp.tid()                                                                         <L 85>
        var_0 = builtin_tid1d();
        // if not env_mask[idx]:                                                                  <L 86>
        var_1 = wp::address(var_env_mask, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::unot(var_3);
        if (var_2) {
            // return                                                                             <L 87>
            continue;
        }
        // out_ang_vel_b[idx] = wp.vec3f(0.0, 0.0, 0.0)                                           <L 89>
        var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
        wp::array_store(var_out_ang_vel_b, var_0, var_7);
        // out_lin_acc_b[idx] = wp.vec3f(0.0, 0.0, 0.0)                                           <L 90>
        var_11 = wp::vec_t<3, wp::float32>(var_8, var_9, var_10);
        wp::array_store(var_out_lin_acc_b, var_0, var_11);
        // prev_lin_vel_w[idx] = wp.vec3f(0.0, 0.0, 0.0)                                          <L 91>
        var_15 = wp::vec_t<3, wp::float32>(var_12, var_13, var_14);
        wp::array_store(var_prev_lin_vel_w, var_0, var_15);
    }
}



extern "C" __global__ void imu_reset_kernel_c1ffcda0_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_ang_vel_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_lin_acc_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_prev_lin_vel_w,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_ang_vel_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_lin_acc_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_prev_lin_vel_w)
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
        const wp::float32 var_6 = 0.0;
        wp::vec_t<3, wp::float32> var_7;
        const wp::float32 var_8 = 0.0;
        const wp::float32 var_9 = 0.0;
        const wp::float32 var_10 = 0.0;
        wp::vec_t<3, wp::float32> var_11;
        const wp::float32 var_12 = 0.0;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        wp::vec_t<3, wp::float32> var_15;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        wp::float32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::vec_t<3, wp::float32> adj_15 = {};
        //---------
        // forward
        // def imu_reset_kernel(                                                                  <L 71>
        // idx = wp.tid()                                                                         <L 85>
        var_0 = builtin_tid1d();
        // if not env_mask[idx]:                                                                  <L 86>
        var_1 = wp::address(var_env_mask, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::unot(var_3);
        if (var_2) {
            // return                                                                             <L 87>
            goto label0;
        }
        // out_ang_vel_b[idx] = wp.vec3f(0.0, 0.0, 0.0)                                           <L 89>
        var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
        // wp::array_store(var_out_ang_vel_b, var_0, var_7);
        // out_lin_acc_b[idx] = wp.vec3f(0.0, 0.0, 0.0)                                           <L 90>
        var_11 = wp::vec_t<3, wp::float32>(var_8, var_9, var_10);
        // wp::array_store(var_out_lin_acc_b, var_0, var_11);
        // prev_lin_vel_w[idx] = wp.vec3f(0.0, 0.0, 0.0)                                          <L 91>
        var_15 = wp::vec_t<3, wp::float32>(var_12, var_13, var_14);
        // wp::array_store(var_prev_lin_vel_w, var_0, var_15);
        //---------
        // reverse
        wp::adj_array_store(var_prev_lin_vel_w, var_0, var_15, adj_prev_lin_vel_w, adj_0, adj_15);
        wp::adj_vec_t(var_12, var_13, var_14, adj_12, adj_13, adj_14, adj_15);
        // adj: prev_lin_vel_w[idx] = wp.vec3f(0.0, 0.0, 0.0)                                     <L 91>
        wp::adj_array_store(var_out_lin_acc_b, var_0, var_11, adj_out_lin_acc_b, adj_0, adj_11);
        wp::adj_vec_t(var_8, var_9, var_10, adj_8, adj_9, adj_10, adj_11);
        // adj: out_lin_acc_b[idx] = wp.vec3f(0.0, 0.0, 0.0)                                      <L 90>
        wp::adj_array_store(var_out_ang_vel_b, var_0, var_7, adj_out_ang_vel_b, adj_0, adj_7);
        wp::adj_vec_t(var_4, var_5, var_6, adj_4, adj_5, adj_6, adj_7);
        // adj: out_ang_vel_b[idx] = wp.vec3f(0.0, 0.0, 0.0)                                      <L 89>
        if (var_2) {
            label0:;
            // adj: return                                                                        <L 87>
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_1);
        // adj: if not env_mask[idx]:                                                             <L 86>
        // adj: idx = wp.tid()                                                                    <L 85>
        // adj: def imu_reset_kernel(                                                             <L 71>
        continue;
    }
}



extern "C" __global__ void imu_update_kernel_79c75563_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_transforms,
    wp::array_t<wp::vec_t<6, wp::float32>> var_velocities,
    wp::array_t<wp::transform_t<wp::float32>> var_coms,
    wp::array_t<wp::vec_t<3, wp::float32>> var_offset_pos_b,
    wp::array_t<wp::quat_t<wp::float32>> var_offset_quat_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity_bias_w,
    wp::float32 var_inv_dt,
    wp::array_t<wp::float32> var_timestamp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_prev_lin_vel_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_ang_vel_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_lin_acc_b)
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
        wp::float32* var_4;
        const wp::float32 var_5 = 0.0;
        bool var_6;
        wp::float32 var_7;
        wp::transform_t<wp::float32>* var_8;
        wp::quat_t<wp::float32> var_9;
        wp::transform_t<wp::float32> var_10;
        wp::vec_t<6, wp::float32>* var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<6, wp::float32> var_13;
        wp::vec_t<6, wp::float32>* var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<6, wp::float32> var_16;
        wp::transform_t<wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::transform_t<wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32>* var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::quat_t<wp::float32>* var_33;
        wp::quat_t<wp::float32> var_34;
        wp::quat_t<wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        //---------
        // forward
        // def imu_update_kernel(                                                                 <L 10>
        // idx = wp.tid()                                                                         <L 43>
        var_0 = builtin_tid1d();
        // if not env_mask[idx]:                                                                  <L 44>
        var_1 = wp::address(var_env_mask, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::unot(var_3);
        if (var_2) {
            // return                                                                             <L 45>
            continue;
        }
        // if timestamp[idx] == 0.0:                                                              <L 49>
        var_4 = wp::address(var_timestamp, var_0);
        var_7 = wp::load(var_4);
        var_6 = (var_7 == var_5);
        if (var_6) {
            // return                                                                             <L 50>
            continue;
        }
        // body_quat = wp.transform_get_rotation(transforms[idx])                                 <L 52>
        var_8 = wp::address(var_transforms, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::transform_get_rotation(var_10);
        // lin_vel_w = wp.spatial_top(velocities[idx])                                            <L 54>
        var_11 = wp::address(var_velocities, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::spatial_top(var_13);
        // ang_vel_w = wp.spatial_bottom(velocities[idx])                                         <L 55>
        var_14 = wp::address(var_velocities, var_0);
        var_16 = wp::load(var_14);
        var_15 = wp::spatial_bottom(var_16);
        // com_pos_b = wp.transform_get_translation(coms[idx])                                    <L 57>
        var_17 = wp::address(var_coms, var_0);
        var_19 = wp::load(var_17);
        var_18 = wp::transform_get_translation(var_19);
        // lever_arm = wp.quat_rotate(body_quat, offset_pos_b[idx] - com_pos_b)                   <L 58>
        var_20 = wp::address(var_offset_pos_b, var_0);
        var_22 = wp::load(var_20);
        var_21 = wp::sub(var_22, var_18);
        var_23 = wp::quat_rotate(var_9, var_21);
        // lin_vel_w = lin_vel_w + wp.cross(ang_vel_w, lever_arm)                                 <L 59>
        var_24 = wp::cross(var_15, var_23);
        var_25 = wp::add(var_12, var_24);
        // lin_acc_w = (lin_vel_w - prev_lin_vel_w[idx]) * inv_dt + gravity_bias_w[idx]           <L 60>
        var_26 = wp::address(var_prev_lin_vel_w, var_0);
        var_28 = wp::load(var_26);
        var_27 = wp::sub(var_25, var_28);
        var_29 = wp::mul(var_27, var_inv_dt);
        var_30 = wp::address(var_gravity_bias_w, var_0);
        var_32 = wp::load(var_30);
        var_31 = wp::add(var_29, var_32);
        // sensor_quat = body_quat * offset_quat_b[idx]                                           <L 62>
        var_33 = wp::address(var_offset_quat_b, var_0);
        var_35 = wp::load(var_33);
        var_34 = wp::mul(var_9, var_35);
        // out_ang_vel_b[idx] = wp.quat_rotate_inv(sensor_quat, ang_vel_w)                        <L 63>
        var_36 = wp::quat_rotate_inv(var_34, var_15);
        wp::array_store(var_out_ang_vel_b, var_0, var_36);
        // out_lin_acc_b[idx] = wp.quat_rotate_inv(sensor_quat, lin_acc_w)                        <L 64>
        var_37 = wp::quat_rotate_inv(var_34, var_31);
        wp::array_store(var_out_lin_acc_b, var_0, var_37);
        // prev_lin_vel_w[idx] = lin_vel_w                                                        <L 67>
        wp::array_store(var_prev_lin_vel_w, var_0, var_25);
    }
}



extern "C" __global__ void imu_update_kernel_79c75563_cuda_kernel_backward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_transforms,
    wp::array_t<wp::vec_t<6, wp::float32>> var_velocities,
    wp::array_t<wp::transform_t<wp::float32>> var_coms,
    wp::array_t<wp::vec_t<3, wp::float32>> var_offset_pos_b,
    wp::array_t<wp::quat_t<wp::float32>> var_offset_quat_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity_bias_w,
    wp::float32 var_inv_dt,
    wp::array_t<wp::float32> var_timestamp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_prev_lin_vel_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_ang_vel_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_lin_acc_b,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_transforms,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_velocities,
    wp::array_t<wp::transform_t<wp::float32>> adj_coms,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_offset_pos_b,
    wp::array_t<wp::quat_t<wp::float32>> adj_offset_quat_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_gravity_bias_w,
    wp::float32 adj_inv_dt,
    wp::array_t<wp::float32> adj_timestamp,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_prev_lin_vel_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_ang_vel_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_lin_acc_b)
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
        wp::float32* var_4;
        const wp::float32 var_5 = 0.0;
        bool var_6;
        wp::float32 var_7;
        wp::transform_t<wp::float32>* var_8;
        wp::quat_t<wp::float32> var_9;
        wp::transform_t<wp::float32> var_10;
        wp::vec_t<6, wp::float32>* var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<6, wp::float32> var_13;
        wp::vec_t<6, wp::float32>* var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<6, wp::float32> var_16;
        wp::transform_t<wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::transform_t<wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32>* var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::quat_t<wp::float32>* var_33;
        wp::quat_t<wp::float32> var_34;
        wp::quat_t<wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        wp::float32 adj_4 = {};
        wp::float32 adj_5 = {};
        bool adj_6 = {};
        wp::float32 adj_7 = {};
        wp::transform_t<wp::float32> adj_8 = {};
        wp::quat_t<wp::float32> adj_9 = {};
        wp::transform_t<wp::float32> adj_10 = {};
        wp::vec_t<6, wp::float32> adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        wp::vec_t<6, wp::float32> adj_13 = {};
        wp::vec_t<6, wp::float32> adj_14 = {};
        wp::vec_t<3, wp::float32> adj_15 = {};
        wp::vec_t<6, wp::float32> adj_16 = {};
        wp::transform_t<wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::transform_t<wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::vec_t<3, wp::float32> adj_29 = {};
        wp::vec_t<3, wp::float32> adj_30 = {};
        wp::vec_t<3, wp::float32> adj_31 = {};
        wp::vec_t<3, wp::float32> adj_32 = {};
        wp::quat_t<wp::float32> adj_33 = {};
        wp::quat_t<wp::float32> adj_34 = {};
        wp::quat_t<wp::float32> adj_35 = {};
        wp::vec_t<3, wp::float32> adj_36 = {};
        wp::vec_t<3, wp::float32> adj_37 = {};
        //---------
        // forward
        // def imu_update_kernel(                                                                 <L 10>
        // idx = wp.tid()                                                                         <L 43>
        var_0 = builtin_tid1d();
        // if not env_mask[idx]:                                                                  <L 44>
        var_1 = wp::address(var_env_mask, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::unot(var_3);
        if (var_2) {
            // return                                                                             <L 45>
            goto label0;
        }
        // if timestamp[idx] == 0.0:                                                              <L 49>
        var_4 = wp::address(var_timestamp, var_0);
        var_7 = wp::load(var_4);
        var_6 = (var_7 == var_5);
        if (var_6) {
            // return                                                                             <L 50>
            goto label1;
        }
        // body_quat = wp.transform_get_rotation(transforms[idx])                                 <L 52>
        var_8 = wp::address(var_transforms, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::transform_get_rotation(var_10);
        // lin_vel_w = wp.spatial_top(velocities[idx])                                            <L 54>
        var_11 = wp::address(var_velocities, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::spatial_top(var_13);
        // ang_vel_w = wp.spatial_bottom(velocities[idx])                                         <L 55>
        var_14 = wp::address(var_velocities, var_0);
        var_16 = wp::load(var_14);
        var_15 = wp::spatial_bottom(var_16);
        // com_pos_b = wp.transform_get_translation(coms[idx])                                    <L 57>
        var_17 = wp::address(var_coms, var_0);
        var_19 = wp::load(var_17);
        var_18 = wp::transform_get_translation(var_19);
        // lever_arm = wp.quat_rotate(body_quat, offset_pos_b[idx] - com_pos_b)                   <L 58>
        var_20 = wp::address(var_offset_pos_b, var_0);
        var_22 = wp::load(var_20);
        var_21 = wp::sub(var_22, var_18);
        var_23 = wp::quat_rotate(var_9, var_21);
        // lin_vel_w = lin_vel_w + wp.cross(ang_vel_w, lever_arm)                                 <L 59>
        var_24 = wp::cross(var_15, var_23);
        var_25 = wp::add(var_12, var_24);
        // lin_acc_w = (lin_vel_w - prev_lin_vel_w[idx]) * inv_dt + gravity_bias_w[idx]           <L 60>
        var_26 = wp::address(var_prev_lin_vel_w, var_0);
        var_28 = wp::load(var_26);
        var_27 = wp::sub(var_25, var_28);
        var_29 = wp::mul(var_27, var_inv_dt);
        var_30 = wp::address(var_gravity_bias_w, var_0);
        var_32 = wp::load(var_30);
        var_31 = wp::add(var_29, var_32);
        // sensor_quat = body_quat * offset_quat_b[idx]                                           <L 62>
        var_33 = wp::address(var_offset_quat_b, var_0);
        var_35 = wp::load(var_33);
        var_34 = wp::mul(var_9, var_35);
        // out_ang_vel_b[idx] = wp.quat_rotate_inv(sensor_quat, ang_vel_w)                        <L 63>
        var_36 = wp::quat_rotate_inv(var_34, var_15);
        // wp::array_store(var_out_ang_vel_b, var_0, var_36);
        // out_lin_acc_b[idx] = wp.quat_rotate_inv(sensor_quat, lin_acc_w)                        <L 64>
        var_37 = wp::quat_rotate_inv(var_34, var_31);
        // wp::array_store(var_out_lin_acc_b, var_0, var_37);
        // prev_lin_vel_w[idx] = lin_vel_w                                                        <L 67>
        // wp::array_store(var_prev_lin_vel_w, var_0, var_25);
        //---------
        // reverse
        wp::adj_array_store(var_prev_lin_vel_w, var_0, var_25, adj_prev_lin_vel_w, adj_0, adj_25);
        // adj: prev_lin_vel_w[idx] = lin_vel_w                                                   <L 67>
        wp::adj_array_store(var_out_lin_acc_b, var_0, var_37, adj_out_lin_acc_b, adj_0, adj_37);
        wp::adj_quat_rotate_inv(var_34, var_31, adj_34, adj_31, adj_37);
        // adj: out_lin_acc_b[idx] = wp.quat_rotate_inv(sensor_quat, lin_acc_w)                   <L 64>
        wp::adj_array_store(var_out_ang_vel_b, var_0, var_36, adj_out_ang_vel_b, adj_0, adj_36);
        wp::adj_quat_rotate_inv(var_34, var_15, adj_34, adj_15, adj_36);
        // adj: out_ang_vel_b[idx] = wp.quat_rotate_inv(sensor_quat, ang_vel_w)                   <L 63>
        wp::adj_mul(var_9, var_35, adj_9, adj_33, adj_34);
        wp::adj_address(var_offset_quat_b, var_0, adj_offset_quat_b, adj_0, adj_33);
        // adj: sensor_quat = body_quat * offset_quat_b[idx]                                      <L 62>
        wp::adj_add(var_29, var_32, adj_29, adj_30, adj_31);
        wp::adj_address(var_gravity_bias_w, var_0, adj_gravity_bias_w, adj_0, adj_30);
        wp::adj_mul(var_27, var_inv_dt, adj_27, adj_inv_dt, adj_29);
        wp::adj_sub(var_25, var_28, adj_25, adj_26, adj_27);
        wp::adj_address(var_prev_lin_vel_w, var_0, adj_prev_lin_vel_w, adj_0, adj_26);
        // adj: lin_acc_w = (lin_vel_w - prev_lin_vel_w[idx]) * inv_dt + gravity_bias_w[idx]      <L 60>
        wp::adj_add(var_12, var_24, adj_12, adj_24, adj_25);
        wp::adj_cross(var_15, var_23, adj_15, adj_23, adj_24);
        // adj: lin_vel_w = lin_vel_w + wp.cross(ang_vel_w, lever_arm)                            <L 59>
        wp::adj_quat_rotate(var_9, var_21, adj_9, adj_21, adj_23);
        wp::adj_sub(var_22, var_18, adj_20, adj_18, adj_21);
        wp::adj_address(var_offset_pos_b, var_0, adj_offset_pos_b, adj_0, adj_20);
        // adj: lever_arm = wp.quat_rotate(body_quat, offset_pos_b[idx] - com_pos_b)              <L 58>
        wp::adj_transform_get_translation(var_19, adj_17, adj_18);
        wp::adj_address(var_coms, var_0, adj_coms, adj_0, adj_17);
        // adj: com_pos_b = wp.transform_get_translation(coms[idx])                               <L 57>
        wp::adj_spatial_bottom(var_16, adj_14, adj_15);
        wp::adj_address(var_velocities, var_0, adj_velocities, adj_0, adj_14);
        // adj: ang_vel_w = wp.spatial_bottom(velocities[idx])                                    <L 55>
        wp::adj_spatial_top(var_13, adj_11, adj_12);
        wp::adj_address(var_velocities, var_0, adj_velocities, adj_0, adj_11);
        // adj: lin_vel_w = wp.spatial_top(velocities[idx])                                       <L 54>
        wp::adj_transform_get_rotation(var_10, adj_8, adj_9);
        wp::adj_address(var_transforms, var_0, adj_transforms, adj_0, adj_8);
        // adj: body_quat = wp.transform_get_rotation(transforms[idx])                            <L 52>
        if (var_6) {
            label1:;
            // adj: return                                                                        <L 50>
        }
        wp::adj_address(var_timestamp, var_0, adj_timestamp, adj_0, adj_4);
        // adj: if timestamp[idx] == 0.0:                                                         <L 49>
        if (var_2) {
            label0:;
            // adj: return                                                                        <L 45>
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_1);
        // adj: if not env_mask[idx]:                                                             <L 44>
        // adj: idx = wp.tid()                                                                    <L 43>
        // adj: def imu_update_kernel(                                                            <L 10>
        continue;
    }
}

