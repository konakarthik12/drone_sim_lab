

def enable_gpu_dynamics():
    from omni.physx import acquire_physx_interface
    physx_interface = acquire_physx_interface()
    physx_interface.overwrite_gpu_setting(1)
