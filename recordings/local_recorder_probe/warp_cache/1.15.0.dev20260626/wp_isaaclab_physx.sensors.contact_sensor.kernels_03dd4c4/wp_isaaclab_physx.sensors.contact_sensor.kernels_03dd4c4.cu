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



extern "C" __global__ void concat_pos_and_quat_to_pose_kernel_e591c4f9_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::transform_t<wp::float32>> var_pose)
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
        wp::int32 var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::quat_t<wp::float32>* var_3;
        wp::transform_t<wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::quat_t<wp::float32> var_6;
        //---------
        // forward
        // def concat_pos_and_quat_to_pose_kernel(                                                <L 284>
        // env, sensor = wp.tid()                                                                 <L 296>
        builtin_tid2d(var_0, var_1);
        // pose[env, sensor] = wp.transform(pos[env, sensor], quat[env, sensor])                  <L 297>
        var_2 = wp::address(var_pos, var_0, var_1);
        var_3 = wp::address(var_quat, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::transform_t<wp::float32>(var_5, var_6);
        wp::array_store(var_pose, var_0, var_1, var_4);
    }
}



extern "C" __global__ void concat_pos_and_quat_to_pose_kernel_e591c4f9_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::transform_t<wp::float32>> var_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_pos,
    wp::array_t<wp::quat_t<wp::float32>> adj_quat,
    wp::array_t<wp::transform_t<wp::float32>> adj_pose)
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
        wp::int32 var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::quat_t<wp::float32>* var_3;
        wp::transform_t<wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::quat_t<wp::float32> var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::quat_t<wp::float32> adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::vec_t<3, wp::float32> adj_5 = {};
        wp::quat_t<wp::float32> adj_6 = {};
        //---------
        // forward
        // def concat_pos_and_quat_to_pose_kernel(                                                <L 284>
        // env, sensor = wp.tid()                                                                 <L 296>
        builtin_tid2d(var_0, var_1);
        // pose[env, sensor] = wp.transform(pos[env, sensor], quat[env, sensor])                  <L 297>
        var_2 = wp::address(var_pos, var_0, var_1);
        var_3 = wp::address(var_quat, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::transform_t<wp::float32>(var_5, var_6);
        // wp::array_store(var_pose, var_0, var_1, var_4);
        //---------
        // reverse
        wp::adj_array_store(var_pose, var_0, var_1, var_4, adj_pose, adj_0, adj_1, adj_4);
        wp::adj_transform_t(var_5, var_6, adj_2, adj_3, adj_4);
        wp::adj_address(var_quat, var_0, var_1, adj_quat, adj_0, adj_1, adj_3);
        wp::adj_address(var_pos, var_0, var_1, adj_pos, adj_0, adj_1, adj_2);
        // adj: pose[env, sensor] = wp.transform(pos[env, sensor], quat[env, sensor])             <L 297>
        // adj: env, sensor = wp.tid()                                                            <L 296>
        // adj: def concat_pos_and_quat_to_pose_kernel(                                           <L 284>
        continue;
    }
}



extern "C" __global__ void reset_contact_sensor_kernel_b98cae5a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_history_length,
    wp::int32 var_num_filter_objects,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix_w,
    wp::array_t<wp::float32> var_current_air_time,
    wp::array_t<wp::float32> var_last_air_time,
    wp::array_t<wp::float32> var_current_contact_time,
    wp::array_t<wp::float32> var_last_contact_time,
    wp::array_t<wp::vec_t<3, wp::float32>> var_friction_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_w)
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
        wp::int32 var_1;
        bool* var_2;
        bool var_3;
        bool var_4;
        const wp::float32 var_5 = 0.0;
        wp::vec_t<3, wp::float32> var_6;
        wp::range_t var_7;
        wp::int32 var_8;
        const wp::float32 var_9 = 0.0;
        wp::vec_t<3, wp::float32> var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        const wp::float32 var_13 = 0.0;
        wp::vec_t<3, wp::float32> var_14;
        const wp::float32 var_15 = 0.0;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        wp::range_t var_19;
        wp::int32 var_20;
        const wp::float32 var_21 = 0.0;
        wp::vec_t<3, wp::float32> var_22;
        wp::int32 var_23;
        wp::range_t var_24;
        wp::int32 var_25;
        const wp::float32 var_26 = 0.0;
        wp::vec_t<3, wp::float32> var_27;
        wp::int32 var_28;
        //---------
        // forward
        // def reset_contact_sensor_kernel(                                                       <L 91>
        // env, sensor = wp.tid()                                                                 <L 126>
        builtin_tid2d(var_0, var_1);
        // if env_mask:                                                                           <L 128>
        if (var_env_mask) {
            // if not env_mask[env]:                                                              <L 129>
            var_2 = wp::address(var_env_mask, var_0);
            var_4 = wp::load(var_2);
            var_3 = wp::unot(var_4);
            if (var_3) {
                // return                                                                         <L 130>
                continue;
            }
        }
        // net_forces_w[env, sensor] = wp.vec3f(0.0)                                              <L 133>
        var_6 = wp::vec_t<3, wp::float32>(var_5);
        wp::array_store(var_net_forces_w, var_0, var_1, var_6);
        // if net_forces_w_history:                                                               <L 136>
        if (var_net_forces_w_history) {
            // for i in range(history_length):                                                    <L 137>
            var_7 = wp::range(var_history_length);
            start_for_1:;
                if (iter_cmp(var_7) == 0) goto end_for_1;
                var_8 = wp::iter_next(var_7);
                // net_forces_w_history[env, i, sensor] = wp.vec3f(0.0)                           <L 138>
                var_10 = wp::vec_t<3, wp::float32>(var_9);
                wp::array_store(var_net_forces_w_history, var_0, var_8, var_1, var_10);
                goto start_for_1;
            end_for_1:;
        }
        // if force_matrix_w:                                                                     <L 141>
        if (var_force_matrix_w) {
            // for f in range(num_filter_objects):                                                <L 142>
            var_11 = wp::range(var_num_filter_objects);
            start_for_3:;
                if (iter_cmp(var_11) == 0) goto end_for_3;
                var_12 = wp::iter_next(var_11);
                // force_matrix_w[env, sensor, f] = wp.vec3f(0.0)                                 <L 143>
                var_14 = wp::vec_t<3, wp::float32>(var_13);
                wp::array_store(var_force_matrix_w, var_0, var_1, var_12, var_14);
                goto start_for_3;
            end_for_3:;
        }
        // if current_air_time:                                                                   <L 146>
        if (var_current_air_time) {
            // current_air_time[env, sensor] = 0.0                                                <L 147>
            wp::array_store(var_current_air_time, var_0, var_1, var_15);
            // last_air_time[env, sensor] = 0.0                                                   <L 148>
            wp::array_store(var_last_air_time, var_0, var_1, var_16);
            // current_contact_time[env, sensor] = 0.0                                            <L 149>
            wp::array_store(var_current_contact_time, var_0, var_1, var_17);
            // last_contact_time[env, sensor] = 0.0                                               <L 150>
            wp::array_store(var_last_contact_time, var_0, var_1, var_18);
        }
        // if friction_forces_w:                                                                  <L 152>
        if (var_friction_forces_w) {
            // for f in range(num_filter_objects):                                                <L 153>
            var_19 = wp::range(var_num_filter_objects);
            start_for_5:;
                if (iter_cmp(var_19) == 0) goto end_for_5;
                var_20 = wp::iter_next(var_19);
                // friction_forces_w[env, sensor, f] = wp.vec3f(0.0)                              <L 154>
                var_22 = wp::vec_t<3, wp::float32>(var_21);
                wp::array_store(var_friction_forces_w, var_0, var_1, var_20, var_22);
                goto start_for_5;
            end_for_5:;
        }
        var_23 = wp::where(var_friction_forces_w, var_20, var_12);
        // if contact_pos_w:                                                                      <L 156>
        if (var_contact_pos_w) {
            // for f in range(num_filter_objects):                                                <L 157>
            var_24 = wp::range(var_num_filter_objects);
            start_for_7:;
                if (iter_cmp(var_24) == 0) goto end_for_7;
                var_25 = wp::iter_next(var_24);
                // contact_pos_w[env, sensor, f] = wp.vec3f(0.0)                                  <L 158>
                var_27 = wp::vec_t<3, wp::float32>(var_26);
                wp::array_store(var_contact_pos_w, var_0, var_1, var_25, var_27);
                goto start_for_7;
            end_for_7:;
        }
        var_28 = wp::where(var_contact_pos_w, var_25, var_23);
    }
}



extern "C" __global__ void reset_contact_sensor_kernel_b98cae5a_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_history_length,
    wp::int32 var_num_filter_objects,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix_w,
    wp::array_t<wp::float32> var_current_air_time,
    wp::array_t<wp::float32> var_last_air_time,
    wp::array_t<wp::float32> var_current_contact_time,
    wp::array_t<wp::float32> var_last_contact_time,
    wp::array_t<wp::vec_t<3, wp::float32>> var_friction_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_w,
    wp::int32 adj_history_length,
    wp::int32 adj_num_filter_objects,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_forces_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_force_matrix_w,
    wp::array_t<wp::float32> adj_current_air_time,
    wp::array_t<wp::float32> adj_last_air_time,
    wp::array_t<wp::float32> adj_current_contact_time,
    wp::array_t<wp::float32> adj_last_contact_time,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_friction_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_contact_pos_w)
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
        wp::int32 var_1;
        bool* var_2;
        bool var_3;
        bool var_4;
        const wp::float32 var_5 = 0.0;
        wp::vec_t<3, wp::float32> var_6;
        wp::range_t var_7;
        wp::int32 var_8;
        const wp::float32 var_9 = 0.0;
        wp::vec_t<3, wp::float32> var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        const wp::float32 var_13 = 0.0;
        wp::vec_t<3, wp::float32> var_14;
        const wp::float32 var_15 = 0.0;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        wp::range_t var_19;
        wp::int32 var_20;
        const wp::float32 var_21 = 0.0;
        wp::vec_t<3, wp::float32> var_22;
        wp::int32 var_23;
        wp::range_t var_24;
        wp::int32 var_25;
        const wp::float32 var_26 = 0.0;
        wp::vec_t<3, wp::float32> var_27;
        wp::int32 var_28;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::float32 adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        wp::range_t adj_7 = {};
        wp::int32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::vec_t<3, wp::float32> adj_10 = {};
        wp::range_t adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::range_t adj_19 = {};
        wp::int32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::int32 adj_23 = {};
        wp::range_t adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::int32 adj_28 = {};
        //---------
        // forward
        // def reset_contact_sensor_kernel(                                                       <L 91>
        // env, sensor = wp.tid()                                                                 <L 126>
        builtin_tid2d(var_0, var_1);
        // if env_mask:                                                                           <L 128>
        if (var_env_mask) {
            // if not env_mask[env]:                                                              <L 129>
            var_2 = wp::address(var_env_mask, var_0);
            var_4 = wp::load(var_2);
            var_3 = wp::unot(var_4);
            if (var_3) {
                // return                                                                         <L 130>
                goto label0;
            }
        }
        // net_forces_w[env, sensor] = wp.vec3f(0.0)                                              <L 133>
        var_6 = wp::vec_t<3, wp::float32>(var_5);
        // wp::array_store(var_net_forces_w, var_0, var_1, var_6);
        // if net_forces_w_history:                                                               <L 136>
        if (var_net_forces_w_history) {
            // for i in range(history_length):                                                    <L 137>
            var_7 = wp::range(var_history_length);
        }
        // if force_matrix_w:                                                                     <L 141>
        if (var_force_matrix_w) {
            // for f in range(num_filter_objects):                                                <L 142>
            var_11 = wp::range(var_num_filter_objects);
        }
        // if current_air_time:                                                                   <L 146>
        if (var_current_air_time) {
            // current_air_time[env, sensor] = 0.0                                                <L 147>
            // wp::array_store(var_current_air_time, var_0, var_1, var_15);
            // last_air_time[env, sensor] = 0.0                                                   <L 148>
            // wp::array_store(var_last_air_time, var_0, var_1, var_16);
            // current_contact_time[env, sensor] = 0.0                                            <L 149>
            // wp::array_store(var_current_contact_time, var_0, var_1, var_17);
            // last_contact_time[env, sensor] = 0.0                                               <L 150>
            // wp::array_store(var_last_contact_time, var_0, var_1, var_18);
        }
        // if friction_forces_w:                                                                  <L 152>
        if (var_friction_forces_w) {
            // for f in range(num_filter_objects):                                                <L 153>
            var_19 = wp::range(var_num_filter_objects);
        }
        var_23 = wp::where(var_friction_forces_w, var_20, var_12);
        // if contact_pos_w:                                                                      <L 156>
        if (var_contact_pos_w) {
            // for f in range(num_filter_objects):                                                <L 157>
            var_24 = wp::range(var_num_filter_objects);
        }
        var_28 = wp::where(var_contact_pos_w, var_25, var_23);
        //---------
        // reverse
        wp::adj_where(var_contact_pos_w, var_25, var_23, adj_contact_pos_w, adj_25, adj_23, adj_28);
        if (var_contact_pos_w) {
            var_24 = wp::iter_reverse(var_24);
            start_for_7:;
                if (iter_cmp(var_24) == 0) goto end_for_7;
                var_25 = wp::iter_next(var_24);
            	adj_26 = {};
            	adj_27 = {};
                // contact_pos_w[env, sensor, f] = wp.vec3f(0.0)                                  <L 158>
                var_27 = wp::vec_t<3, wp::float32>(var_26);
                // wp::array_store(var_contact_pos_w, var_0, var_1, var_25, var_27);
                wp::adj_array_store(var_contact_pos_w, var_0, var_1, var_25, var_27, adj_contact_pos_w, adj_0, adj_1, adj_25, adj_27);
                wp::adj_vec_t(var_26, adj_26, adj_27);
                // adj: contact_pos_w[env, sensor, f] = wp.vec3f(0.0)                             <L 158>
            	goto start_for_7;
            end_for_7:;
            // adj: for f in range(num_filter_objects):                                           <L 157>
        }
        // adj: if contact_pos_w:                                                                 <L 156>
        wp::adj_where(var_friction_forces_w, var_20, var_12, adj_friction_forces_w, adj_20, adj_12, adj_23);
        if (var_friction_forces_w) {
            var_19 = wp::iter_reverse(var_19);
            start_for_5:;
                if (iter_cmp(var_19) == 0) goto end_for_5;
                var_20 = wp::iter_next(var_19);
            	adj_21 = {};
            	adj_22 = {};
                // friction_forces_w[env, sensor, f] = wp.vec3f(0.0)                              <L 154>
                var_22 = wp::vec_t<3, wp::float32>(var_21);
                // wp::array_store(var_friction_forces_w, var_0, var_1, var_20, var_22);
                wp::adj_array_store(var_friction_forces_w, var_0, var_1, var_20, var_22, adj_friction_forces_w, adj_0, adj_1, adj_20, adj_22);
                wp::adj_vec_t(var_21, adj_21, adj_22);
                // adj: friction_forces_w[env, sensor, f] = wp.vec3f(0.0)                         <L 154>
            	goto start_for_5;
            end_for_5:;
            // adj: for f in range(num_filter_objects):                                           <L 153>
        }
        // adj: if friction_forces_w:                                                             <L 152>
        if (var_current_air_time) {
            wp::adj_array_store(var_last_contact_time, var_0, var_1, var_18, adj_last_contact_time, adj_0, adj_1, adj_18);
            // adj: last_contact_time[env, sensor] = 0.0                                          <L 150>
            wp::adj_array_store(var_current_contact_time, var_0, var_1, var_17, adj_current_contact_time, adj_0, adj_1, adj_17);
            // adj: current_contact_time[env, sensor] = 0.0                                       <L 149>
            wp::adj_array_store(var_last_air_time, var_0, var_1, var_16, adj_last_air_time, adj_0, adj_1, adj_16);
            // adj: last_air_time[env, sensor] = 0.0                                              <L 148>
            wp::adj_array_store(var_current_air_time, var_0, var_1, var_15, adj_current_air_time, adj_0, adj_1, adj_15);
            // adj: current_air_time[env, sensor] = 0.0                                           <L 147>
        }
        // adj: if current_air_time:                                                              <L 146>
        if (var_force_matrix_w) {
            var_11 = wp::iter_reverse(var_11);
            start_for_3:;
                if (iter_cmp(var_11) == 0) goto end_for_3;
                var_12 = wp::iter_next(var_11);
            	adj_13 = {};
            	adj_14 = {};
                // force_matrix_w[env, sensor, f] = wp.vec3f(0.0)                                 <L 143>
                var_14 = wp::vec_t<3, wp::float32>(var_13);
                // wp::array_store(var_force_matrix_w, var_0, var_1, var_12, var_14);
                wp::adj_array_store(var_force_matrix_w, var_0, var_1, var_12, var_14, adj_force_matrix_w, adj_0, adj_1, adj_12, adj_14);
                wp::adj_vec_t(var_13, adj_13, adj_14);
                // adj: force_matrix_w[env, sensor, f] = wp.vec3f(0.0)                            <L 143>
            	goto start_for_3;
            end_for_3:;
            // adj: for f in range(num_filter_objects):                                           <L 142>
        }
        // adj: if force_matrix_w:                                                                <L 141>
        if (var_net_forces_w_history) {
            var_7 = wp::iter_reverse(var_7);
            start_for_1:;
                if (iter_cmp(var_7) == 0) goto end_for_1;
                var_8 = wp::iter_next(var_7);
            	adj_9 = {};
            	adj_10 = {};
                // net_forces_w_history[env, i, sensor] = wp.vec3f(0.0)                           <L 138>
                var_10 = wp::vec_t<3, wp::float32>(var_9);
                // wp::array_store(var_net_forces_w_history, var_0, var_8, var_1, var_10);
                wp::adj_array_store(var_net_forces_w_history, var_0, var_8, var_1, var_10, adj_net_forces_w_history, adj_0, adj_8, adj_1, adj_10);
                wp::adj_vec_t(var_9, adj_9, adj_10);
                // adj: net_forces_w_history[env, i, sensor] = wp.vec3f(0.0)                      <L 138>
            	goto start_for_1;
            end_for_1:;
            // adj: for i in range(history_length):                                               <L 137>
        }
        // adj: if net_forces_w_history:                                                          <L 136>
        wp::adj_array_store(var_net_forces_w, var_0, var_1, var_6, adj_net_forces_w, adj_0, adj_1, adj_6);
        wp::adj_vec_t(var_5, adj_5, adj_6);
        // adj: net_forces_w[env, sensor] = wp.vec3f(0.0)                                         <L 133>
        if (var_env_mask) {
            if (var_3) {
                label0:;
                // adj: return                                                                    <L 130>
            }
            wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_2);
            // adj: if not env_mask[env]:                                                         <L 129>
        }
        // adj: if env_mask:                                                                      <L 128>
        // adj: env, sensor = wp.tid()                                                            <L 126>
        // adj: def reset_contact_sensor_kernel(                                                  <L 91>
        continue;
    }
}



extern "C" __global__ void update_net_forces_kernel_2bb591c7_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_flat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_matrix_flat,
    wp::array_t<bool> var_mask,
    wp::int32 var_num_sensors,
    wp::int32 var_num_filter_shapes,
    wp::int32 var_history_length,
    wp::float32 var_contact_force_threshold,
    wp::array_t<wp::float32> var_timestamp,
    wp::array_t<wp::float32> var_timestamp_last_update,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix_w_history,
    wp::array_t<wp::float32> var_current_air_time,
    wp::array_t<wp::float32> var_current_contact_time,
    wp::array_t<wp::float32> var_last_air_time,
    wp::array_t<wp::float32> var_last_contact_time)
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
        wp::int32 var_1;
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::float32* var_5;
        const wp::float32 var_6 = 0.0;
        bool var_7;
        wp::float32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<3, wp::float32>* var_11;
        wp::vec_t<3, wp::float32> var_12;
        const wp::int32 var_13 = 1;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        const wp::int32 var_16 = -1;
        wp::range_t var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32>* var_23;
        const wp::int32 var_24 = 0;
        wp::vec_t<3, wp::float32> var_25;
        wp::range_t var_26;
        wp::int32 var_27;
        wp::vec_t<3, wp::float32>* var_28;
        wp::vec_t<3, wp::float32> var_29;
        const wp::int32 var_30 = 1;
        wp::int32 var_31;
        const wp::int32 var_32 = 0;
        const wp::int32 var_33 = -1;
        wp::range_t var_34;
        wp::int32 var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        wp::vec_t<3, wp::float32>* var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32>* var_40;
        const wp::int32 var_41 = 0;
        wp::vec_t<3, wp::float32> var_42;
        wp::float32* var_43;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        wp::vec_t<3, wp::float32>* var_48;
        wp::float32 var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::float32 var_51;
        bool var_52;
        wp::float32* var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::float32* var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        bool var_59;
        const wp::float32 var_60 = 0.0;
        bool var_61;
        bool var_62;
        bool var_63;
        const wp::float32 var_64 = 0.0;
        bool var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        const wp::float32 var_69 = 0.0;
        wp::float32 var_70;
        const wp::float32 var_71 = 0.0;
        wp::float32 var_72;
        wp::float32 var_73;
        //---------
        // forward
        // def update_net_forces_kernel(                                                          <L 190>
        // env, sensor = wp.tid()                                                                 <L 235>
        builtin_tid2d(var_0, var_1);
        // if mask:                                                                               <L 237>
        if (var_mask) {
            // if not mask[env]:                                                                  <L 238>
            var_2 = wp::address(var_mask, var_0);
            var_4 = wp::load(var_2);
            var_3 = wp::unot(var_4);
            if (var_3) {
                // return                                                                         <L 239>
                continue;
            }
        }
        // if timestamp[env] == 0.0:                                                              <L 243>
        var_5 = wp::address(var_timestamp, var_0);
        var_8 = wp::load(var_5);
        var_7 = (var_8 == var_6);
        if (var_7) {
            // return                                                                             <L 244>
            continue;
        }
        // src_idx = env * num_sensors + sensor                                                   <L 246>
        var_9 = wp::mul(var_0, var_num_sensors);
        var_10 = wp::add(var_9, var_1);
        // net_forces_w[env, sensor] = net_forces_flat[src_idx]                                   <L 249>
        var_11 = wp::address(var_net_forces_flat, var_10);
        var_12 = wp::load(var_11);
        wp::array_store(var_net_forces_w, var_0, var_1, var_12);
        // if net_forces_w_history:                                                               <L 251>
        if (var_net_forces_w_history) {
            // for i in range(history_length - 1, 0, -1):                                         <L 252>
            var_14 = wp::sub(var_history_length, var_13);
            var_17 = wp::range(var_14, var_15, var_16);
            start_for_2:;
                if (iter_cmp(var_17) == 0) goto end_for_2;
                var_18 = wp::iter_next(var_17);
                // net_forces_w_history[env, i, sensor] = net_forces_w_history[env, i - 1, sensor]       <L 253>
                var_20 = wp::sub(var_18, var_19);
                var_21 = wp::address(var_net_forces_w_history, var_0, var_20, var_1);
                var_22 = wp::load(var_21);
                wp::array_store(var_net_forces_w_history, var_0, var_18, var_1, var_22);
                goto start_for_2;
            end_for_2:;
            // net_forces_w_history[env, 0, sensor] = net_forces_w[env, sensor]                   <L 254>
            var_23 = wp::address(var_net_forces_w, var_0, var_1);
            var_25 = wp::load(var_23);
            wp::array_store(var_net_forces_w_history, var_0, var_24, var_1, var_25);
        }
        // if net_forces_matrix_flat:                                                             <L 257>
        if (var_net_forces_matrix_flat) {
            // for f in range(num_filter_shapes):                                                 <L 258>
            var_26 = wp::range(var_num_filter_shapes);
            start_for_4:;
                if (iter_cmp(var_26) == 0) goto end_for_4;
                var_27 = wp::iter_next(var_26);
                // force_matrix_w[env, sensor, f] = net_forces_matrix_flat[src_idx, f]            <L 259>
                var_28 = wp::address(var_net_forces_matrix_flat, var_10, var_27);
                var_29 = wp::load(var_28);
                wp::array_store(var_force_matrix_w, var_0, var_1, var_27, var_29);
                // for i in range(history_length - 1, 0, -1):                                     <L 260>
                var_31 = wp::sub(var_history_length, var_30);
                var_34 = wp::range(var_31, var_32, var_33);
                start_for_6:;
                    if (iter_cmp(var_34) == 0) goto end_for_6;
                    var_35 = wp::iter_next(var_34);
                    // force_matrix_w_history[env, i, sensor, f] = force_matrix_w_history[env, i - 1, sensor, f]       <L 261>
                    var_37 = wp::sub(var_35, var_36);
                    var_38 = wp::address(var_force_matrix_w_history, var_0, var_37, var_1, var_27);
                    var_39 = wp::load(var_38);
                    wp::array_store(var_force_matrix_w_history, var_0, var_35, var_1, var_27, var_39);
                    goto start_for_6;
                end_for_6:;
                // force_matrix_w_history[env, 0, sensor, f] = force_matrix_w[env, sensor, f]       <L 262>
                var_40 = wp::address(var_force_matrix_w, var_0, var_1, var_27);
                var_42 = wp::load(var_40);
                wp::array_store(var_force_matrix_w_history, var_0, var_41, var_1, var_27, var_42);
                wp::assign(var_18, var_35);
                goto start_for_4;
            end_for_4:;
        }
        // if current_air_time:                                                                   <L 265>
        if (var_current_air_time) {
            // elapsed_time = timestamp[env] - timestamp_last_update[env]                         <L 266>
            var_43 = wp::address(var_timestamp, var_0);
            var_44 = wp::address(var_timestamp_last_update, var_0);
            var_46 = wp::load(var_43);
            var_47 = wp::load(var_44);
            var_45 = wp::sub(var_46, var_47);
            // in_contact = wp.length_sq(net_forces_w[env, sensor]) > contact_force_threshold * contact_force_threshold       <L 267>
            var_48 = wp::address(var_net_forces_w, var_0, var_1);
            var_50 = wp::load(var_48);
            var_49 = wp::length_sq(var_50);
            var_51 = wp::mul(var_contact_force_threshold, var_contact_force_threshold);
            var_52 = (var_49 > var_51);
            // cat = current_air_time[env, sensor]                                                <L 269>
            var_53 = wp::address(var_current_air_time, var_0, var_1);
            var_55 = wp::load(var_53);
            var_54 = wp::copy(var_55);
            // cct = current_contact_time[env, sensor]                                            <L 270>
            var_56 = wp::address(var_current_contact_time, var_0, var_1);
            var_58 = wp::load(var_56);
            var_57 = wp::copy(var_58);
            // is_first_contact = in_contact and (cat > 0.0)                                      <L 271>
            var_59 = var_52;
            if (var_59) {
                var_61 = (var_54 > var_60);
                var_59 = var_59 && var_61;
            }
            // is_first_detached = not in_contact and (cct > 0.0)                                 <L 272>
            var_63 = wp::unot(var_52);
            var_62 = var_63;
            if (var_62) {
                var_65 = (var_57 > var_64);
                var_62 = var_62 && var_65;
            }
            // if is_first_contact:                                                               <L 274>
            if (var_59) {
                // last_air_time[env, sensor] = cat + elapsed_time                                <L 275>
                var_66 = wp::add(var_54, var_45);
                wp::array_store(var_last_air_time, var_0, var_1, var_66);
            }
            if (!var_59) {
                // elif is_first_detached:                                                        <L 276>
                if (var_62) {
                    // last_contact_time[env, sensor] = cct + elapsed_time                        <L 277>
                    var_67 = wp::add(var_57, var_45);
                    wp::array_store(var_last_contact_time, var_0, var_1, var_67);
                }
            }
            // current_contact_time[env, sensor] = wp.where(in_contact, cct + elapsed_time, 0.0)       <L 279>
            var_68 = wp::add(var_57, var_45);
            var_70 = wp::where(var_52, var_68, var_69);
            wp::array_store(var_current_contact_time, var_0, var_1, var_70);
            // current_air_time[env, sensor] = wp.where(in_contact, 0.0, cat + elapsed_time)       <L 280>
            var_72 = wp::add(var_54, var_45);
            var_73 = wp::where(var_52, var_71, var_72);
            wp::array_store(var_current_air_time, var_0, var_1, var_73);
        }
    }
}



extern "C" __global__ void update_net_forces_kernel_2bb591c7_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_flat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_matrix_flat,
    wp::array_t<bool> var_mask,
    wp::int32 var_num_sensors,
    wp::int32 var_num_filter_shapes,
    wp::int32 var_history_length,
    wp::float32 var_contact_force_threshold,
    wp::array_t<wp::float32> var_timestamp,
    wp::array_t<wp::float32> var_timestamp_last_update,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_net_forces_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_force_matrix_w_history,
    wp::array_t<wp::float32> var_current_air_time,
    wp::array_t<wp::float32> var_current_contact_time,
    wp::array_t<wp::float32> var_last_air_time,
    wp::array_t<wp::float32> var_last_contact_time,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_forces_flat,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_forces_matrix_flat,
    wp::array_t<bool> adj_mask,
    wp::int32 adj_num_sensors,
    wp::int32 adj_num_filter_shapes,
    wp::int32 adj_history_length,
    wp::float32 adj_contact_force_threshold,
    wp::array_t<wp::float32> adj_timestamp,
    wp::array_t<wp::float32> adj_timestamp_last_update,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_forces_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_net_forces_w_history,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_force_matrix_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_force_matrix_w_history,
    wp::array_t<wp::float32> adj_current_air_time,
    wp::array_t<wp::float32> adj_current_contact_time,
    wp::array_t<wp::float32> adj_last_air_time,
    wp::array_t<wp::float32> adj_last_contact_time)
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
        wp::int32 var_1;
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::float32* var_5;
        const wp::float32 var_6 = 0.0;
        bool var_7;
        wp::float32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<3, wp::float32>* var_11;
        wp::vec_t<3, wp::float32> var_12;
        const wp::int32 var_13 = 1;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        const wp::int32 var_16 = -1;
        wp::range_t var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32>* var_23;
        const wp::int32 var_24 = 0;
        wp::vec_t<3, wp::float32> var_25;
        wp::range_t var_26;
        wp::int32 var_27;
        wp::vec_t<3, wp::float32>* var_28;
        wp::vec_t<3, wp::float32> var_29;
        const wp::int32 var_30 = 1;
        wp::int32 var_31;
        const wp::int32 var_32 = 0;
        const wp::int32 var_33 = -1;
        wp::range_t var_34;
        wp::int32 var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        wp::vec_t<3, wp::float32>* var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32>* var_40;
        const wp::int32 var_41 = 0;
        wp::vec_t<3, wp::float32> var_42;
        wp::float32* var_43;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        wp::vec_t<3, wp::float32>* var_48;
        wp::float32 var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::float32 var_51;
        bool var_52;
        wp::float32* var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::float32* var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        bool var_59;
        const wp::float32 var_60 = 0.0;
        bool var_61;
        bool var_62;
        bool var_63;
        const wp::float32 var_64 = 0.0;
        bool var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        const wp::float32 var_69 = 0.0;
        wp::float32 var_70;
        const wp::float32 var_71 = 0.0;
        wp::float32 var_72;
        wp::float32 var_73;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        bool adj_7 = {};
        wp::float32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::range_t adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::int32 adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::range_t adj_26 = {};
        wp::int32 adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::vec_t<3, wp::float32> adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::range_t adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::vec_t<3, wp::float32> adj_38 = {};
        wp::vec_t<3, wp::float32> adj_39 = {};
        wp::vec_t<3, wp::float32> adj_40 = {};
        wp::int32 adj_41 = {};
        wp::vec_t<3, wp::float32> adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::vec_t<3, wp::float32> adj_48 = {};
        wp::float32 adj_49 = {};
        wp::vec_t<3, wp::float32> adj_50 = {};
        wp::float32 adj_51 = {};
        bool adj_52 = {};
        wp::float32 adj_53 = {};
        wp::float32 adj_54 = {};
        wp::float32 adj_55 = {};
        wp::float32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::float32 adj_58 = {};
        bool adj_59 = {};
        wp::float32 adj_60 = {};
        bool adj_61 = {};
        bool adj_62 = {};
        bool adj_63 = {};
        wp::float32 adj_64 = {};
        bool adj_65 = {};
        wp::float32 adj_66 = {};
        wp::float32 adj_67 = {};
        wp::float32 adj_68 = {};
        wp::float32 adj_69 = {};
        wp::float32 adj_70 = {};
        wp::float32 adj_71 = {};
        wp::float32 adj_72 = {};
        wp::float32 adj_73 = {};
        //---------
        // forward
        // def update_net_forces_kernel(                                                          <L 190>
        // env, sensor = wp.tid()                                                                 <L 235>
        builtin_tid2d(var_0, var_1);
        // if mask:                                                                               <L 237>
        if (var_mask) {
            // if not mask[env]:                                                                  <L 238>
            var_2 = wp::address(var_mask, var_0);
            var_4 = wp::load(var_2);
            var_3 = wp::unot(var_4);
            if (var_3) {
                // return                                                                         <L 239>
                goto label0;
            }
        }
        // if timestamp[env] == 0.0:                                                              <L 243>
        var_5 = wp::address(var_timestamp, var_0);
        var_8 = wp::load(var_5);
        var_7 = (var_8 == var_6);
        if (var_7) {
            // return                                                                             <L 244>
            goto label1;
        }
        // src_idx = env * num_sensors + sensor                                                   <L 246>
        var_9 = wp::mul(var_0, var_num_sensors);
        var_10 = wp::add(var_9, var_1);
        // net_forces_w[env, sensor] = net_forces_flat[src_idx]                                   <L 249>
        var_11 = wp::address(var_net_forces_flat, var_10);
        var_12 = wp::load(var_11);
        // wp::array_store(var_net_forces_w, var_0, var_1, var_12);
        // if net_forces_w_history:                                                               <L 251>
        if (var_net_forces_w_history) {
            // for i in range(history_length - 1, 0, -1):                                         <L 252>
            var_14 = wp::sub(var_history_length, var_13);
            var_17 = wp::range(var_14, var_15, var_16);
            // net_forces_w_history[env, 0, sensor] = net_forces_w[env, sensor]                   <L 254>
            var_23 = wp::address(var_net_forces_w, var_0, var_1);
            var_25 = wp::load(var_23);
            // wp::array_store(var_net_forces_w_history, var_0, var_24, var_1, var_25);
        }
        // if net_forces_matrix_flat:                                                             <L 257>
        if (var_net_forces_matrix_flat) {
            // for f in range(num_filter_shapes):                                                 <L 258>
            var_26 = wp::range(var_num_filter_shapes);
        }
        // if current_air_time:                                                                   <L 265>
        if (var_current_air_time) {
            // elapsed_time = timestamp[env] - timestamp_last_update[env]                         <L 266>
            var_43 = wp::address(var_timestamp, var_0);
            var_44 = wp::address(var_timestamp_last_update, var_0);
            var_46 = wp::load(var_43);
            var_47 = wp::load(var_44);
            var_45 = wp::sub(var_46, var_47);
            // in_contact = wp.length_sq(net_forces_w[env, sensor]) > contact_force_threshold * contact_force_threshold       <L 267>
            var_48 = wp::address(var_net_forces_w, var_0, var_1);
            var_50 = wp::load(var_48);
            var_49 = wp::length_sq(var_50);
            var_51 = wp::mul(var_contact_force_threshold, var_contact_force_threshold);
            var_52 = (var_49 > var_51);
            // cat = current_air_time[env, sensor]                                                <L 269>
            var_53 = wp::address(var_current_air_time, var_0, var_1);
            var_55 = wp::load(var_53);
            var_54 = wp::copy(var_55);
            // cct = current_contact_time[env, sensor]                                            <L 270>
            var_56 = wp::address(var_current_contact_time, var_0, var_1);
            var_58 = wp::load(var_56);
            var_57 = wp::copy(var_58);
            // is_first_contact = in_contact and (cat > 0.0)                                      <L 271>
            var_59 = var_52;
            if (var_59) {
                var_61 = (var_54 > var_60);
                var_59 = var_59 && var_61;
            }
            // is_first_detached = not in_contact and (cct > 0.0)                                 <L 272>
            var_63 = wp::unot(var_52);
            var_62 = var_63;
            if (var_62) {
                var_65 = (var_57 > var_64);
                var_62 = var_62 && var_65;
            }
            // if is_first_contact:                                                               <L 274>
            if (var_59) {
                // last_air_time[env, sensor] = cat + elapsed_time                                <L 275>
                var_66 = wp::add(var_54, var_45);
                // wp::array_store(var_last_air_time, var_0, var_1, var_66);
            }
            if (!var_59) {
                // elif is_first_detached:                                                        <L 276>
                if (var_62) {
                    // last_contact_time[env, sensor] = cct + elapsed_time                        <L 277>
                    var_67 = wp::add(var_57, var_45);
                    // wp::array_store(var_last_contact_time, var_0, var_1, var_67);
                }
            }
            // current_contact_time[env, sensor] = wp.where(in_contact, cct + elapsed_time, 0.0)       <L 279>
            var_68 = wp::add(var_57, var_45);
            var_70 = wp::where(var_52, var_68, var_69);
            // wp::array_store(var_current_contact_time, var_0, var_1, var_70);
            // current_air_time[env, sensor] = wp.where(in_contact, 0.0, cat + elapsed_time)       <L 280>
            var_72 = wp::add(var_54, var_45);
            var_73 = wp::where(var_52, var_71, var_72);
            // wp::array_store(var_current_air_time, var_0, var_1, var_73);
        }
        //---------
        // reverse
        if (var_current_air_time) {
            wp::adj_array_store(var_current_air_time, var_0, var_1, var_73, adj_current_air_time, adj_0, adj_1, adj_73);
            wp::adj_where(var_52, var_71, var_72, adj_52, adj_71, adj_72, adj_73);
            wp::adj_add(var_54, var_45, adj_54, adj_45, adj_72);
            // adj: current_air_time[env, sensor] = wp.where(in_contact, 0.0, cat + elapsed_time)  <L 280>
            wp::adj_array_store(var_current_contact_time, var_0, var_1, var_70, adj_current_contact_time, adj_0, adj_1, adj_70);
            wp::adj_where(var_52, var_68, var_69, adj_52, adj_68, adj_69, adj_70);
            wp::adj_add(var_57, var_45, adj_57, adj_45, adj_68);
            // adj: current_contact_time[env, sensor] = wp.where(in_contact, cct + elapsed_time, 0.0)  <L 279>
            if (!var_59) {
                if (var_62) {
                    wp::adj_array_store(var_last_contact_time, var_0, var_1, var_67, adj_last_contact_time, adj_0, adj_1, adj_67);
                    wp::adj_add(var_57, var_45, adj_57, adj_45, adj_67);
                    // adj: last_contact_time[env, sensor] = cct + elapsed_time                   <L 277>
                }
                // adj: elif is_first_detached:                                                   <L 276>
            }
            if (var_59) {
                wp::adj_array_store(var_last_air_time, var_0, var_1, var_66, adj_last_air_time, adj_0, adj_1, adj_66);
                wp::adj_add(var_54, var_45, adj_54, adj_45, adj_66);
                // adj: last_air_time[env, sensor] = cat + elapsed_time                           <L 275>
            }
            // adj: if is_first_contact:                                                          <L 274>
            if (var_62) {
            }
            // adj: is_first_detached = not in_contact and (cct > 0.0)                            <L 272>
            if (var_59) {
            }
            // adj: is_first_contact = in_contact and (cat > 0.0)                                 <L 271>
            wp::adj_copy(var_58, adj_56, adj_57);
            wp::adj_address(var_current_contact_time, var_0, var_1, adj_current_contact_time, adj_0, adj_1, adj_56);
            // adj: cct = current_contact_time[env, sensor]                                       <L 270>
            wp::adj_copy(var_55, adj_53, adj_54);
            wp::adj_address(var_current_air_time, var_0, var_1, adj_current_air_time, adj_0, adj_1, adj_53);
            // adj: cat = current_air_time[env, sensor]                                           <L 269>
            wp::adj_mul(var_contact_force_threshold, var_contact_force_threshold, adj_contact_force_threshold, adj_contact_force_threshold, adj_51);
            wp::adj_length_sq(var_50, adj_48, adj_49);
            wp::adj_address(var_net_forces_w, var_0, var_1, adj_net_forces_w, adj_0, adj_1, adj_48);
            // adj: in_contact = wp.length_sq(net_forces_w[env, sensor]) > contact_force_threshold * contact_force_threshold  <L 267>
            wp::adj_sub(var_46, var_47, adj_43, adj_44, adj_45);
            wp::adj_address(var_timestamp_last_update, var_0, adj_timestamp_last_update, adj_0, adj_44);
            wp::adj_address(var_timestamp, var_0, adj_timestamp, adj_0, adj_43);
            // adj: elapsed_time = timestamp[env] - timestamp_last_update[env]                    <L 266>
        }
        // adj: if current_air_time:                                                              <L 265>
        if (var_net_forces_matrix_flat) {
            var_26 = wp::iter_reverse(var_26);
            start_for_4:;
                if (iter_cmp(var_26) == 0) goto end_for_4;
                var_27 = wp::iter_next(var_26);
            	adj_28 = {};
            	adj_29 = {};
            	adj_30 = {};
            	adj_31 = {};
            	adj_32 = {};
            	adj_33 = {};
            	adj_34 = {};
            	adj_40 = {};
            	adj_41 = {};
            	adj_42 = {};
                // force_matrix_w[env, sensor, f] = net_forces_matrix_flat[src_idx, f]            <L 259>
                var_28 = wp::address(var_net_forces_matrix_flat, var_10, var_27);
                var_29 = wp::load(var_28);
                // wp::array_store(var_force_matrix_w, var_0, var_1, var_27, var_29);
                // for i in range(history_length - 1, 0, -1):                                     <L 260>
                var_31 = wp::sub(var_history_length, var_30);
                var_34 = wp::range(var_31, var_32, var_33);
                // force_matrix_w_history[env, 0, sensor, f] = force_matrix_w[env, sensor, f]       <L 262>
                var_40 = wp::address(var_force_matrix_w, var_0, var_1, var_27);
                var_42 = wp::load(var_40);
                // wp::array_store(var_force_matrix_w_history, var_0, var_41, var_1, var_27, var_42);
                wp::assign(var_18, var_35);
                wp::adj_assign(var_18, var_35, adj_18, adj_35);
                wp::adj_array_store(var_force_matrix_w_history, var_0, var_41, var_1, var_27, var_42, adj_force_matrix_w_history, adj_0, adj_41, adj_1, adj_27, adj_40);
                wp::adj_address(var_force_matrix_w, var_0, var_1, var_27, adj_force_matrix_w, adj_0, adj_1, adj_27, adj_40);
                // adj: force_matrix_w_history[env, 0, sensor, f] = force_matrix_w[env, sensor, f]  <L 262>
                var_34 = wp::iter_reverse(var_34);
                start_for_6:;
                    if (iter_cmp(var_34) == 0) goto end_for_6;
                    var_35 = wp::iter_next(var_34);
                	adj_36 = {};
                	adj_37 = {};
                	adj_38 = {};
                	adj_39 = {};
                    // force_matrix_w_history[env, i, sensor, f] = force_matrix_w_history[env, i - 1, sensor, f]       <L 261>
                    var_37 = wp::sub(var_35, var_36);
                    var_38 = wp::address(var_force_matrix_w_history, var_0, var_37, var_1, var_27);
                    var_39 = wp::load(var_38);
                    // wp::array_store(var_force_matrix_w_history, var_0, var_35, var_1, var_27, var_39);
                    wp::adj_array_store(var_force_matrix_w_history, var_0, var_35, var_1, var_27, var_39, adj_force_matrix_w_history, adj_0, adj_35, adj_1, adj_27, adj_38);
                    wp::adj_address(var_force_matrix_w_history, var_0, var_37, var_1, var_27, adj_force_matrix_w_history, adj_0, adj_37, adj_1, adj_27, adj_38);
                    wp::adj_sub(var_35, var_36, adj_35, adj_36, adj_37);
                    // adj: force_matrix_w_history[env, i, sensor, f] = force_matrix_w_history[env, i - 1, sensor, f]  <L 261>
                	goto start_for_6;
                end_for_6:;
                wp::adj_sub(var_history_length, var_30, adj_history_length, adj_30, adj_31);
                // adj: for i in range(history_length - 1, 0, -1):                                <L 260>
                wp::adj_array_store(var_force_matrix_w, var_0, var_1, var_27, var_29, adj_force_matrix_w, adj_0, adj_1, adj_27, adj_28);
                wp::adj_address(var_net_forces_matrix_flat, var_10, var_27, adj_net_forces_matrix_flat, adj_10, adj_27, adj_28);
                // adj: force_matrix_w[env, sensor, f] = net_forces_matrix_flat[src_idx, f]       <L 259>
            	goto start_for_4;
            end_for_4:;
            // adj: for f in range(num_filter_shapes):                                            <L 258>
        }
        // adj: if net_forces_matrix_flat:                                                        <L 257>
        if (var_net_forces_w_history) {
            wp::adj_array_store(var_net_forces_w_history, var_0, var_24, var_1, var_25, adj_net_forces_w_history, adj_0, adj_24, adj_1, adj_23);
            wp::adj_address(var_net_forces_w, var_0, var_1, adj_net_forces_w, adj_0, adj_1, adj_23);
            // adj: net_forces_w_history[env, 0, sensor] = net_forces_w[env, sensor]              <L 254>
            var_17 = wp::iter_reverse(var_17);
            start_for_2:;
                if (iter_cmp(var_17) == 0) goto end_for_2;
                var_18 = wp::iter_next(var_17);
            	adj_19 = {};
            	adj_20 = {};
            	adj_21 = {};
            	adj_22 = {};
                // net_forces_w_history[env, i, sensor] = net_forces_w_history[env, i - 1, sensor]       <L 253>
                var_20 = wp::sub(var_18, var_19);
                var_21 = wp::address(var_net_forces_w_history, var_0, var_20, var_1);
                var_22 = wp::load(var_21);
                // wp::array_store(var_net_forces_w_history, var_0, var_18, var_1, var_22);
                wp::adj_array_store(var_net_forces_w_history, var_0, var_18, var_1, var_22, adj_net_forces_w_history, adj_0, adj_18, adj_1, adj_21);
                wp::adj_address(var_net_forces_w_history, var_0, var_20, var_1, adj_net_forces_w_history, adj_0, adj_20, adj_1, adj_21);
                wp::adj_sub(var_18, var_19, adj_18, adj_19, adj_20);
                // adj: net_forces_w_history[env, i, sensor] = net_forces_w_history[env, i - 1, sensor]  <L 253>
            	goto start_for_2;
            end_for_2:;
            wp::adj_sub(var_history_length, var_13, adj_history_length, adj_13, adj_14);
            // adj: for i in range(history_length - 1, 0, -1):                                    <L 252>
        }
        // adj: if net_forces_w_history:                                                          <L 251>
        wp::adj_array_store(var_net_forces_w, var_0, var_1, var_12, adj_net_forces_w, adj_0, adj_1, adj_11);
        wp::adj_address(var_net_forces_flat, var_10, adj_net_forces_flat, adj_10, adj_11);
        // adj: net_forces_w[env, sensor] = net_forces_flat[src_idx]                              <L 249>
        wp::adj_add(var_9, var_1, adj_9, adj_1, adj_10);
        wp::adj_mul(var_0, var_num_sensors, adj_0, adj_num_sensors, adj_9);
        // adj: src_idx = env * num_sensors + sensor                                              <L 246>
        if (var_7) {
            label1:;
            // adj: return                                                                        <L 244>
        }
        wp::adj_address(var_timestamp, var_0, adj_timestamp, adj_0, adj_5);
        // adj: if timestamp[env] == 0.0:                                                         <L 243>
        if (var_mask) {
            if (var_3) {
                label0:;
                // adj: return                                                                    <L 239>
            }
            wp::adj_address(var_mask, var_0, adj_mask, adj_0, adj_2);
            // adj: if not mask[env]:                                                             <L 238>
        }
        // adj: if mask:                                                                          <L 237>
        // adj: env, sensor = wp.tid()                                                            <L 235>
        // adj: def update_net_forces_kernel(                                                     <L 190>
        continue;
    }
}



extern "C" __global__ void split_flat_pose_to_pos_quat_5486bfa0_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_src,
    wp::array_t<bool> var_mask,
    wp::int32 var_num_bodies,
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_dst_quat)
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
        wp::int32 var_1;
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::transform_t<wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::quat_t<wp::float32> var_11;
        wp::transform_t<wp::float32> var_12;
        //---------
        // forward
        // def split_flat_pose_to_pos_quat(                                                       <L 14>
        // env, sensor = wp.tid()                                                                 <L 30>
        builtin_tid2d(var_0, var_1);
        // if mask:                                                                               <L 31>
        if (var_mask) {
            // if not mask[env]:                                                                  <L 32>
            var_2 = wp::address(var_mask, var_0);
            var_4 = wp::load(var_2);
            var_3 = wp::unot(var_4);
            if (var_3) {
                // return                                                                         <L 33>
                continue;
            }
        }
        // src_idx = env * num_bodies + sensor                                                    <L 35>
        var_5 = wp::mul(var_0, var_num_bodies);
        var_6 = wp::add(var_5, var_1);
        // dst_pos[env, sensor] = wp.transform_get_translation(src[src_idx])                      <L 36>
        var_7 = wp::address(var_src, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::transform_get_translation(var_9);
        wp::array_store(var_dst_pos, var_0, var_1, var_8);
        // dst_quat[env, sensor] = wp.transform_get_rotation(src[src_idx])                        <L 37>
        var_10 = wp::address(var_src, var_6);
        var_12 = wp::load(var_10);
        var_11 = wp::transform_get_rotation(var_12);
        wp::array_store(var_dst_quat, var_0, var_1, var_11);
    }
}



extern "C" __global__ void split_flat_pose_to_pos_quat_5486bfa0_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::transform_t<wp::float32>> var_src,
    wp::array_t<bool> var_mask,
    wp::int32 var_num_bodies,
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_dst_quat,
    wp::array_t<wp::transform_t<wp::float32>> adj_src,
    wp::array_t<bool> adj_mask,
    wp::int32 adj_num_bodies,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_dst_pos,
    wp::array_t<wp::quat_t<wp::float32>> adj_dst_quat)
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
        wp::int32 var_1;
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::transform_t<wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::transform_t<wp::float32> var_9;
        wp::transform_t<wp::float32>* var_10;
        wp::quat_t<wp::float32> var_11;
        wp::transform_t<wp::float32> var_12;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::transform_t<wp::float32> adj_9 = {};
        wp::transform_t<wp::float32> adj_10 = {};
        wp::quat_t<wp::float32> adj_11 = {};
        wp::transform_t<wp::float32> adj_12 = {};
        //---------
        // forward
        // def split_flat_pose_to_pos_quat(                                                       <L 14>
        // env, sensor = wp.tid()                                                                 <L 30>
        builtin_tid2d(var_0, var_1);
        // if mask:                                                                               <L 31>
        if (var_mask) {
            // if not mask[env]:                                                                  <L 32>
            var_2 = wp::address(var_mask, var_0);
            var_4 = wp::load(var_2);
            var_3 = wp::unot(var_4);
            if (var_3) {
                // return                                                                         <L 33>
                goto label0;
            }
        }
        // src_idx = env * num_bodies + sensor                                                    <L 35>
        var_5 = wp::mul(var_0, var_num_bodies);
        var_6 = wp::add(var_5, var_1);
        // dst_pos[env, sensor] = wp.transform_get_translation(src[src_idx])                      <L 36>
        var_7 = wp::address(var_src, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::transform_get_translation(var_9);
        // wp::array_store(var_dst_pos, var_0, var_1, var_8);
        // dst_quat[env, sensor] = wp.transform_get_rotation(src[src_idx])                        <L 37>
        var_10 = wp::address(var_src, var_6);
        var_12 = wp::load(var_10);
        var_11 = wp::transform_get_rotation(var_12);
        // wp::array_store(var_dst_quat, var_0, var_1, var_11);
        //---------
        // reverse
        wp::adj_array_store(var_dst_quat, var_0, var_1, var_11, adj_dst_quat, adj_0, adj_1, adj_11);
        wp::adj_transform_get_rotation(var_12, adj_10, adj_11);
        wp::adj_address(var_src, var_6, adj_src, adj_6, adj_10);
        // adj: dst_quat[env, sensor] = wp.transform_get_rotation(src[src_idx])                   <L 37>
        wp::adj_array_store(var_dst_pos, var_0, var_1, var_8, adj_dst_pos, adj_0, adj_1, adj_8);
        wp::adj_transform_get_translation(var_9, adj_7, adj_8);
        wp::adj_address(var_src, var_6, adj_src, adj_6, adj_7);
        // adj: dst_pos[env, sensor] = wp.transform_get_translation(src[src_idx])                 <L 36>
        wp::adj_add(var_5, var_1, adj_5, adj_1, adj_6);
        wp::adj_mul(var_0, var_num_bodies, adj_0, adj_num_bodies, adj_5);
        // adj: src_idx = env * num_bodies + sensor                                               <L 35>
        if (var_mask) {
            if (var_3) {
                label0:;
                // adj: return                                                                    <L 33>
            }
            wp::adj_address(var_mask, var_0, adj_mask, adj_0, adj_2);
            // adj: if not mask[env]:                                                             <L 32>
        }
        // adj: if mask:                                                                          <L 31>
        // adj: env, sensor = wp.tid()                                                            <L 30>
        // adj: def split_flat_pose_to_pos_quat(                                                  <L 14>
        continue;
    }
}



extern "C" __global__ void unpack_contact_buffer_data_c9092c30_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_data,
    wp::array_t<wp::uint32> var_buffer_count,
    wp::array_t<wp::uint32> var_buffer_start_indices,
    wp::array_t<bool> var_mask,
    wp::int32 var_num_bodies,
    bool var_avg,
    wp::float32 var_default_val,
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst)
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
        wp::int32 var_1;
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        bool var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::uint32* var_8;
        wp::int32 var_9;
        wp::uint32 var_10;
        wp::uint32* var_11;
        wp::int32 var_12;
        wp::uint32 var_13;
        const wp::int32 var_14 = 0;
        bool var_15;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        wp::vec_t<3, wp::float32> var_19;
        wp::range_t var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32>* var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::float32 var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        //---------
        // forward
        // def unpack_contact_buffer_data(                                                        <L 44>
        // env, sensor, contact = wp.tid()                                                        <L 70>
        builtin_tid3d(var_0, var_1, var_2);
        // if mask:                                                                               <L 71>
        if (var_mask) {
            // if not mask[env]:                                                                  <L 72>
            var_3 = wp::address(var_mask, var_0);
            var_5 = wp::load(var_3);
            var_4 = wp::unot(var_5);
            if (var_4) {
                // return                                                                         <L 73>
                continue;
            }
        }
        // flat_idx = env * num_bodies + sensor                                                   <L 75>
        var_6 = wp::mul(var_0, var_num_bodies);
        var_7 = wp::add(var_6, var_1);
        // count = wp.int32(buffer_count[flat_idx, contact])                                      <L 76>
        var_8 = wp::address(var_buffer_count, var_7, var_2);
        var_10 = wp::load(var_8);
        var_9 = wp::int32(var_10);
        // start = wp.int32(buffer_start_indices[flat_idx, contact])                              <L 77>
        var_11 = wp::address(var_buffer_start_indices, var_7, var_2);
        var_13 = wp::load(var_11);
        var_12 = wp::int32(var_13);
        // if count > 0:                                                                          <L 79>
        var_15 = (var_9 > var_14);
        if (var_15) {
            // accum = wp.vec3f(0.0, 0.0, 0.0)                                                    <L 80>
            var_19 = wp::vec_t<3, wp::float32>(var_16, var_17, var_18);
            // for c in range(count):                                                             <L 81>
            var_20 = wp::range(var_9);
            start_for_1:;
                if (iter_cmp(var_20) == 0) goto end_for_1;
                var_21 = wp::iter_next(var_20);
                // accum = accum + contact_data[start + c]                                        <L 82>
                var_22 = wp::add(var_12, var_21);
                var_23 = wp::address(var_contact_data, var_22);
                var_25 = wp::load(var_23);
                var_24 = wp::add(var_19, var_25);
                wp::assign(var_19, var_24);
                goto start_for_1;
            end_for_1:;
            // if avg:                                                                            <L 83>
            if (var_avg) {
                // accum = accum / wp.float32(count)                                              <L 84>
                var_26 = wp::float32(var_9);
                var_27 = wp::div(var_19, var_26);
            }
            var_28 = wp::where(var_avg, var_27, var_19);
            // dst[env, sensor, contact] = accum                                                  <L 85>
            wp::array_store(var_dst, var_0, var_1, var_2, var_28);
        }
        if (!var_15) {
            // dst[env, sensor, contact] = wp.vec3f(default_val, default_val, default_val)        <L 87>
            var_29 = wp::vec_t<3, wp::float32>(var_default_val, var_default_val, var_default_val);
            wp::array_store(var_dst, var_0, var_1, var_2, var_29);
        }
    }
}



extern "C" __global__ void unpack_contact_buffer_data_c9092c30_cuda_kernel_backward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_data,
    wp::array_t<wp::uint32> var_buffer_count,
    wp::array_t<wp::uint32> var_buffer_start_indices,
    wp::array_t<bool> var_mask,
    wp::int32 var_num_bodies,
    bool var_avg,
    wp::float32 var_default_val,
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_contact_data,
    wp::array_t<wp::uint32> adj_buffer_count,
    wp::array_t<wp::uint32> adj_buffer_start_indices,
    wp::array_t<bool> adj_mask,
    wp::int32 adj_num_bodies,
    bool adj_avg,
    wp::float32 adj_default_val,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_dst)
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
        wp::int32 var_1;
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        bool var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::uint32* var_8;
        wp::int32 var_9;
        wp::uint32 var_10;
        wp::uint32* var_11;
        wp::int32 var_12;
        wp::uint32 var_13;
        const wp::int32 var_14 = 0;
        bool var_15;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        wp::vec_t<3, wp::float32> var_19;
        wp::range_t var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32>* var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::float32 var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::uint32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::uint32 adj_10 = {};
        wp::uint32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::uint32 adj_13 = {};
        wp::int32 adj_14 = {};
        bool adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::range_t adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::float32 adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::vec_t<3, wp::float32> adj_29 = {};
        //---------
        // forward
        // def unpack_contact_buffer_data(                                                        <L 44>
        // env, sensor, contact = wp.tid()                                                        <L 70>
        builtin_tid3d(var_0, var_1, var_2);
        // if mask:                                                                               <L 71>
        if (var_mask) {
            // if not mask[env]:                                                                  <L 72>
            var_3 = wp::address(var_mask, var_0);
            var_5 = wp::load(var_3);
            var_4 = wp::unot(var_5);
            if (var_4) {
                // return                                                                         <L 73>
                goto label0;
            }
        }
        // flat_idx = env * num_bodies + sensor                                                   <L 75>
        var_6 = wp::mul(var_0, var_num_bodies);
        var_7 = wp::add(var_6, var_1);
        // count = wp.int32(buffer_count[flat_idx, contact])                                      <L 76>
        var_8 = wp::address(var_buffer_count, var_7, var_2);
        var_10 = wp::load(var_8);
        var_9 = wp::int32(var_10);
        // start = wp.int32(buffer_start_indices[flat_idx, contact])                              <L 77>
        var_11 = wp::address(var_buffer_start_indices, var_7, var_2);
        var_13 = wp::load(var_11);
        var_12 = wp::int32(var_13);
        // if count > 0:                                                                          <L 79>
        var_15 = (var_9 > var_14);
        if (var_15) {
            // accum = wp.vec3f(0.0, 0.0, 0.0)                                                    <L 80>
            var_19 = wp::vec_t<3, wp::float32>(var_16, var_17, var_18);
            // for c in range(count):                                                             <L 81>
            var_20 = wp::range(var_9);
            // if avg:                                                                            <L 83>
            if (var_avg) {
                // accum = accum / wp.float32(count)                                              <L 84>
                var_26 = wp::float32(var_9);
                var_27 = wp::div(var_19, var_26);
            }
            var_28 = wp::where(var_avg, var_27, var_19);
            // dst[env, sensor, contact] = accum                                                  <L 85>
            // wp::array_store(var_dst, var_0, var_1, var_2, var_28);
        }
        if (!var_15) {
            // dst[env, sensor, contact] = wp.vec3f(default_val, default_val, default_val)        <L 87>
            var_29 = wp::vec_t<3, wp::float32>(var_default_val, var_default_val, var_default_val);
            // wp::array_store(var_dst, var_0, var_1, var_2, var_29);
        }
        //---------
        // reverse
        if (!var_15) {
            wp::adj_array_store(var_dst, var_0, var_1, var_2, var_29, adj_dst, adj_0, adj_1, adj_2, adj_29);
            wp::adj_vec_t(var_default_val, var_default_val, var_default_val, adj_default_val, adj_default_val, adj_default_val, adj_29);
            // adj: dst[env, sensor, contact] = wp.vec3f(default_val, default_val, default_val)   <L 87>
        }
        if (var_15) {
            wp::adj_array_store(var_dst, var_0, var_1, var_2, var_28, adj_dst, adj_0, adj_1, adj_2, adj_28);
            // adj: dst[env, sensor, contact] = accum                                             <L 85>
            wp::adj_where(var_avg, var_27, var_19, adj_avg, adj_27, adj_19, adj_28);
            if (var_avg) {
                wp::adj_div(var_19, var_26, adj_19, adj_26, adj_27);
                wp::adj_float32(var_9, adj_9, adj_26);
                // adj: accum = accum / wp.float32(count)                                         <L 84>
            }
            // adj: if avg:                                                                       <L 83>
            var_20 = wp::iter_reverse(var_20);
            start_for_1:;
                if (iter_cmp(var_20) == 0) goto end_for_1;
                var_21 = wp::iter_next(var_20);
            	adj_22 = {};
            	adj_23 = {};
            	adj_24 = {};
            	adj_25 = {};
                // accum = accum + contact_data[start + c]                                        <L 82>
                var_22 = wp::add(var_12, var_21);
                var_23 = wp::address(var_contact_data, var_22);
                var_25 = wp::load(var_23);
                var_24 = wp::add(var_19, var_25);
                wp::assign(var_19, var_24);
                wp::adj_assign(var_19, var_24, adj_19, adj_24);
                wp::adj_add(var_19, var_25, adj_19, adj_23, adj_24);
                wp::adj_address(var_contact_data, var_22, adj_contact_data, adj_22, adj_23);
                wp::adj_add(var_12, var_21, adj_12, adj_21, adj_22);
                // adj: accum = accum + contact_data[start + c]                                   <L 82>
            	goto start_for_1;
            end_for_1:;
            // adj: for c in range(count):                                                        <L 81>
            wp::adj_vec_t(var_16, var_17, var_18, adj_16, adj_17, adj_18, adj_19);
            // adj: accum = wp.vec3f(0.0, 0.0, 0.0)                                               <L 80>
        }
        // adj: if count > 0:                                                                     <L 79>
        wp::adj_address(var_buffer_start_indices, var_7, var_2, adj_buffer_start_indices, adj_7, adj_2, adj_11);
        // adj: start = wp.int32(buffer_start_indices[flat_idx, contact])                         <L 77>
        wp::adj_address(var_buffer_count, var_7, var_2, adj_buffer_count, adj_7, adj_2, adj_8);
        // adj: count = wp.int32(buffer_count[flat_idx, contact])                                 <L 76>
        wp::adj_add(var_6, var_1, adj_6, adj_1, adj_7);
        wp::adj_mul(var_0, var_num_bodies, adj_0, adj_num_bodies, adj_6);
        // adj: flat_idx = env * num_bodies + sensor                                              <L 75>
        if (var_mask) {
            if (var_4) {
                label0:;
                // adj: return                                                                    <L 73>
            }
            wp::adj_address(var_mask, var_0, adj_mask, adj_0, adj_3);
            // adj: if not mask[env]:                                                             <L 72>
        }
        // adj: if mask:                                                                          <L 71>
        // adj: env, sensor, contact = wp.tid()                                                   <L 70>
        // adj: def unpack_contact_buffer_data(                                                   <L 44>
        continue;
    }
}



extern "C" __global__ void compute_first_transition_kernel_0adcd6c5_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_threshold,
    wp::array_t<wp::float32> var_time,
    wp::array_t<wp::float32> var_result)
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
        wp::int32 var_1;
        wp::float32* var_2;
        wp::float32 var_3;
        wp::float32 var_4;
        bool var_5;
        const wp::float32 var_6 = 0.0;
        bool var_7;
        bool var_8;
        const wp::float32 var_9 = 1.0;
        const wp::float32 var_10 = 0.0;
        //---------
        // forward
        // def compute_first_transition_kernel(                                                   <L 162>
        // env, sensor = wp.tid()                                                                 <L 181>
        builtin_tid2d(var_0, var_1);
        // t = time[env, sensor]                                                                  <L 182>
        var_2 = wp::address(var_time, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if t > 0.0 and t < threshold:                                                          <L 183>
        var_7 = (var_3 > var_6);
        var_5 = var_7;
        if (var_5) {
            var_8 = (var_3 < var_threshold);
            var_5 = var_5 && var_8;
        }
        if (var_5) {
            // result[env, sensor] = 1.0                                                          <L 184>
            wp::array_store(var_result, var_0, var_1, var_9);
        }
        if (!var_5) {
            // result[env, sensor] = 0.0                                                          <L 186>
            wp::array_store(var_result, var_0, var_1, var_10);
        }
    }
}



extern "C" __global__ void compute_first_transition_kernel_0adcd6c5_cuda_kernel_backward(
    wp::launch_bounds_t<2> dim,
    wp::float32 var_threshold,
    wp::array_t<wp::float32> var_time,
    wp::array_t<wp::float32> var_result,
    wp::float32 adj_threshold,
    wp::array_t<wp::float32> adj_time,
    wp::array_t<wp::float32> adj_result)
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
        wp::int32 var_1;
        wp::float32* var_2;
        wp::float32 var_3;
        wp::float32 var_4;
        bool var_5;
        const wp::float32 var_6 = 0.0;
        bool var_7;
        bool var_8;
        const wp::float32 var_9 = 1.0;
        const wp::float32 var_10 = 0.0;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::float32 adj_4 = {};
        bool adj_5 = {};
        wp::float32 adj_6 = {};
        bool adj_7 = {};
        bool adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        //---------
        // forward
        // def compute_first_transition_kernel(                                                   <L 162>
        // env, sensor = wp.tid()                                                                 <L 181>
        builtin_tid2d(var_0, var_1);
        // t = time[env, sensor]                                                                  <L 182>
        var_2 = wp::address(var_time, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if t > 0.0 and t < threshold:                                                          <L 183>
        var_7 = (var_3 > var_6);
        var_5 = var_7;
        if (var_5) {
            var_8 = (var_3 < var_threshold);
            var_5 = var_5 && var_8;
        }
        if (var_5) {
            // result[env, sensor] = 1.0                                                          <L 184>
            // wp::array_store(var_result, var_0, var_1, var_9);
        }
        if (!var_5) {
            // result[env, sensor] = 0.0                                                          <L 186>
            // wp::array_store(var_result, var_0, var_1, var_10);
        }
        //---------
        // reverse
        if (!var_5) {
            wp::adj_array_store(var_result, var_0, var_1, var_10, adj_result, adj_0, adj_1, adj_10);
            // adj: result[env, sensor] = 0.0                                                     <L 186>
        }
        if (var_5) {
            wp::adj_array_store(var_result, var_0, var_1, var_9, adj_result, adj_0, adj_1, adj_9);
            // adj: result[env, sensor] = 1.0                                                     <L 184>
        }
        if (var_5) {
        }
        // adj: if t > 0.0 and t < threshold:                                                     <L 183>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_time, var_0, var_1, adj_time, adj_0, adj_1, adj_2);
        // adj: t = time[env, sensor]                                                             <L 182>
        // adj: env, sensor = wp.tid()                                                            <L 181>
        // adj: def compute_first_transition_kernel(                                              <L 162>
        continue;
    }
}

