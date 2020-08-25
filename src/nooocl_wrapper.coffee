{
  CLHost
  CLContext
  CLCommandQueue
  CLBuffer
  CLError
  NDRange
} = nooocl = require "nooocl"
host = CLHost.createV11()
# host = CLHost.createV12()
defs = host.cl.defs

# fix
Promise = require "bluebird"
Promise.prototype.cb = (cb)->
  @catch (err)=>cb err
  @then (res)=>cb null, res 

module.exports = 
  device_list_get : ()->
    device_list = []
    
    for platform in host.getPlatforms()
      device_list.append platform.gpuDevices()
    
    device_list
  
  device_ctx : (device)->
    new CLContext device
  
  # ###################################################################################################
  #    queue
  # ###################################################################################################
  ctx_device_queue : (ctx, device)->
    queue = new CLCommandQueue ctx, device
  
  ctx_device_queue_out_of_order : (ctx, device)->
    queue = new CLCommandQueue ctx, device, true
  
  # ###################################################################################################
  ctx_buffer_alloc : (ctx, size, host_buf)->
    if host_buf
      throw new Error "size != host_buf.length; #{size} != #{host_buf.length}" if size != host_buf.length
      new CLBuffer ctx, defs.CL_MEM_READ_WRITE | defs.CL_MEM_USE_HOST_PTR, size, host_buf
    else
      new CLBuffer ctx, defs.CL_MEM_READ_WRITE, size
  # ###################################################################################################
  #    queue slow (but benchmarkable)
  # ###################################################################################################
  queue_kernel : (queue, kernel, local_size, global_size, on_end)->
    loc = new NDRange local_size
    glb = new NDRange global_size
    queue.waitable().enqueueNDRangeKernel(kernel, glb, loc).promise.cb on_end
  
  queue_d2h : (queue, gpu_buf, host_buf, size, on_end)->
    queue.waitable().enqueueReadBuffer(gpu_buf, 0, size, host_buf).promise.cb on_end
  
  queue_h2d : (queue, gpu_buf, host_buf, size, on_end)->
    queue.waitable().enqueueWriteBuffer(gpu_buf, 0, size, host_buf).promise.cb on_end
  
  # ###################################################################################################
  #    queue fast
  # ###################################################################################################
  queue_kernel_fast : (queue, kernel, local_size, global_size, flush)->
    loc = new NDRange local_size
    glb = new NDRange global_size
    queue.enqueueNDRangeKernel(kernel, glb, loc)
    queue.flush() if flush
    return
  
  queue_d2h_fast : (queue, gpu_buf, host_buf, size, flush)->
    queue.enqueueReadBuffer(gpu_buf, 0, size, host_buf)
    queue.flush() if flush
    return
  
  queue_h2d_fast : (queue, gpu_buf, host_buf, size, flush)->
    queue.enqueueWriteBuffer(gpu_buf, 0, size, host_buf)
    queue.flush() if flush
    return
  
  queue_finish : (queue, on_end)->
    queue.finish()
    on_end()
  
  # не работает
  # queue_finish : (queue, on_end)->
    # queue.waitable().finish().promise.cb on_end
  
  # ###################################################################################################
  #    queue chaining
  # ###################################################################################################
  # BUG events + flush ... doesn't work (last wait does not wait)
  queue_kernel_chain : (queue, kernel, local_size, global_size, wait_event_list, flush)->
    loc = new NDRange local_size
    glb = new NDRange global_size
    event = queue.waitable().enqueueNDRangeKernel(kernel, glb, loc, null, wait_event_list)
    # queue.flush() if flush
    event
  
  queue_d2h_chain : (queue, gpu_buf, host_buf, size, wait_event_list, flush)->
    event = queue.waitable().enqueueReadBuffer(gpu_buf, 0, size, host_buf, wait_event_list)
    # queue.flush() if flush
    event
  
  queue_h2d_chain : (queue, gpu_buf, host_buf, size, wait_event_list, flush)->
    event = queue.waitable().enqueueWriteBuffer(gpu_buf, 0, size, host_buf, wait_event_list)
    # queue.flush() if flush
    event
  
  event_wait : (event, on_end)->
    event.promise.cb on_end
  
  # ###################################################################################################
  #    kernel
  # ###################################################################################################
  ctx_device_kernel_build : (ctx, device, kernel_source_code, opt, cb)->
    program = ctx.createProgram kernel_source_code
    
    # TODO use opt
    await program.build("-cl-fast-relaxed-math").then defer()
    build_status= program.getBuildStatus device
    build_log   = program.getBuildLog device
    perr build_log if build_log
    if build_status < 0
      return cb new CLError build_status, "Build failed."
    
    kernel = program.createKernel "_main"
    # _last_program = program
    cb null, kernel
  
  kernel_set_arg : (kernel, num, val, type)->
    kernel.setArg num, val, type
  
  # kernel_set_args : (kernel, list)->
    # kernel.setArgs list...
