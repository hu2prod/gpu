module = @
require "fy"
fs = require "fs"
now = require "performance-now"
gpu_wrapper = require "./nooocl_wrapper"
# ###################################################################################################
# ooo == out of order
@device_list_get = ()->
  _device_list = gpu_wrapper.device_list_get()
  device_list = []
  for _device in _device_list
    device_list.push device = new module.GPU_device
    device._device = _device
  
  device_list
  

class @GPU_device
  _device : null
  ctx : ()->
    ret = new module.GPU_ctx
    ret.parent_device = @
    ret._ctx = gpu_wrapper.device_ctx @_device
    ret

class @GPU_ctx
  parent_device : null
  _ctx : null
  queue : ()->
    ret = new module.GPU_queue
    ret.parent_ctx = @
    ret._queue = gpu_wrapper.ctx_device_queue @_ctx, @parent_device._device
    ret
  
  queue_ooo : ()->
    ret = new module.GPU_queue_ooo
    ret.parent_ctx = @
    ret._queue = gpu_wrapper.ctx_device_queue_out_of_order @_ctx, @parent_device._device
    ret
  
  buffer : (size)->
    ret = new module.GPU_buffer
    ret.parent_ctx = @
    ret.init {size}
    ret
  
  buffer_host : (size)->
    ret = new module.GPU_buffer
    ret.parent_ctx = @
    ret.init {
      size
      is_host : true
    }
    ret
  
  buffer_device : (size)->
    ret = new module.GPU_buffer
    ret.parent_ctx = @
    ret.init {
      size
      host_skip : true
    }
    ret
  
  kernel_code : (code, warp_size, worker_count, cb)->
    return cb new Error "!isFinite warp_size"     if !isFinite warp_size
    return cb new Error "!isFinite worker_count"  if !isFinite worker_count
    
    return cb new Error "warp_size <= 0"    if warp_size <= 0
    return cb new Error "worker_count<= 0"  if worker_count <= 0
    
    code = """
      typedef          char   i8;
      typedef unsigned char   u8;
      typedef          short  i16;
      typedef unsigned short  u16;
      typedef          int    i32;
      typedef unsigned int    u32;
      typedef          long   i64;
      typedef unsigned long   u64;
      typedef          float  f32;
      typedef          double f64;
      
      
      typedef        char4    i8x4;
      typedef        char8    i8x8;
      typedef        char16   i8x16;
      
      typedef        uchar4   u8x4;
      typedef        uchar8   u8x8;
      typedef        uchar16  u8x16;
      
      
      typedef        short4   i16x4;
      typedef        short8   i16x8;
      typedef        short16  i16x16;
      
      typedef        ushort4  u16x4;
      typedef        ushort8  u16x8;
      typedef        ushort16 u16x16;
      
      
      typedef        int4     i32x4;
      typedef        int8     i32x8;
      typedef        int16    i32x16;
      
      typedef        uint4    u32x4;
      typedef        uint8    u32x8;
      typedef        uint16   u32x16;
      
      
      typedef        long4    i64x4;
      typedef        long8    i64x8;
      typedef        long16   i64x16;
      
      typedef        ulong4   u64x4;
      typedef        ulong8   u64x8;
      typedef        ulong16  u64x16;
      
      
      typedef        float4   f32x4;
      typedef        float8   f32x8;
      typedef        float16  f32x16;
      
      typedef        double4  f64x4;
      typedef        double8  f64x8;
      typedef        double16 f64x16;
      
      
      #{code}
      """
    await gpu_wrapper.ctx_device_kernel_build @_ctx, @parent_device._device, code, {}, defer(err, _kernel); return cb err if err
    ret = new module.GPU_kernel
    ret.parent_ctx  = @
    ret._kernel     = _kernel
    ret.warp_size   = warp_size
    ret.worker_count= worker_count
    ret.code        = code
    cb null, ret
  
  kernel_path : (path, warp_size, worker_count, cb)->
    await fs.readFile path, defer(err, buf); return cb err if err
    @kernel_code buf, warp_size, worker_count, cb

class @GPU_queue
  parent_ctx: null
  _queue    : null
  FASTER    : false
  FLUSH     : true
  
  kernel : (kernel, on_end)->
    {
      warp_size
      worker_count
    } = kernel
    worker_count = Math.ceil(worker_count / warp_size) * warp_size
    if @FASTER
      gpu_wrapper.queue_kernel_fast @_queue, kernel._kernel, warp_size, worker_count, @FLUSH
    else
      start_ts = now()
      await gpu_wrapper.queue_kernel @_queue, kernel._kernel, warp_size, worker_count, defer(err); return on_end err if err
      kernel._bench now() - start_ts
    on_end null
  
  d2h : (buffer, on_end)->
    return on_end() if buffer.is_host
    if !buffer.host
      return on_end new Error "d2h ERROR: gpu_buffer is host_skip (device only). You can't move it"
    if @FASTER
      gpu_wrapper.queue_d2h_fast @_queue, buffer._device_buf, buffer.host, buffer.size, @FLUSH
    else
      start_ts = now()
      await gpu_wrapper.queue_d2h @_queue, buffer._device_buf, buffer.host, buffer.size, defer(err); return on_end err if err
      buffer._bench_d2h now() - start_ts
    on_end null
  
  h2d : (buffer, on_end)->
    return on_end() if buffer.is_host
    if !buffer.host
      return on_end new Error "h2d ERROR: gpu_buffer is host_skip (device only). You can't move it"
    if @FASTER
      gpu_wrapper.queue_h2d_fast @_queue, buffer._device_buf, buffer.host, buffer.size, @FLUSH
    else
      start_ts = now()
      await gpu_wrapper.queue_h2d @_queue, buffer._device_buf, buffer.host, buffer.size, defer(err); return on_end err if err
      buffer._bench_h2d now() - start_ts
    on_end null
  
  finish : (on_end)->
    if @FASTER
      gpu_wrapper.queue_finish @_queue, on_end
    else
      on_end()
  
  fork : ()->@

class @GPU_queue_ooo
  parent_ctx: null
  _queue    : null
  last_event: null
  FLUSH     : true
  
  kernel : (kernel, on_end)->
    {
      warp_size
      worker_count
    } = kernel
    worker_count = Math.ceil(worker_count / warp_size) * warp_size
    
    event_list = []
    event_list.push @last_event if @last_event
    # p "kernel event_list", event_list
    @last_event = gpu_wrapper.queue_kernel_chain @_queue, kernel._kernel, warp_size, worker_count, event_list, @FLUSH
    # p "@last_event", [@last_event]
    on_end null
  
  d2h : (buffer, on_end)->
    return on_end() if buffer.is_host
    if !buffer.host
      return on_end new Error "d2h ERROR: gpu_buffer is host_skip (device only). You can't move it"
    
    event_list = []
    event_list.push @last_event if @last_event
    # p "d2h", event_list
    @last_event = gpu_wrapper.queue_d2h_chain @_queue, buffer._device_buf, buffer.host, buffer.size, event_list, @FLUSH
    # p "@last_event", [@last_event]
    on_end null
  
  h2d : (buffer, on_end)->
    return on_end() if buffer.is_host
    if !buffer.host
      return on_end new Error "h2d ERROR: gpu_buffer is host_skip (device only). You can't move it"
    
    event_list = []
    event_list.push @last_event if @last_event
    # p "h2d", event_list
    @last_event = gpu_wrapper.queue_h2d_chain @_queue, buffer._device_buf, buffer.host, buffer.size, event_list, @FLUSH
    # p "@last_event", [@last_event]
    on_end null
  
  finish : (on_end)->
    await gpu_wrapper.event_wait @last_event, defer(err); return on_end err if err
    @last_event = null
    on_end()
  
  fork : ()->
    ret = new module.GPU_queue_ooo
    ret.parent_ctx= @parent_ctx
    ret._queue    = @_queue
    ret.last_event= @last_event
    ret.FLUSH     = @FLUSH
    ret

class @GPU_buffer
  parent_ctx  : null
  
  host        : null
  _device_buf : null
  
  size        : 0
  is_host     : false
  
  init : (opt = {})->
    throw new Error "!opt.size?"    if !opt.size?
    throw new Error "opt.size <= 0" if opt.size <= 0
    throw new Error "opt.is_host and opt.host_skip" if opt.is_host and opt.host_skip
    
    @size    = opt.size
    @is_host = opt.is_host ? false
    if !opt.host_skip
      @host = Buffer.alloc @size
    
    if @is_host
      @_device_buf = gpu_wrapper.ctx_buffer_alloc @parent_ctx._ctx, @size, @host
    else
      @_device_buf = gpu_wrapper.ctx_buffer_alloc @parent_ctx._ctx, @size
    
    return
  
  last_d2h_elp_ts : -1
  last_h2d_elp_ts : -1
  
  _bench_d2h : (@last_d2h_elp_ts)->
  _bench_h2d : (@last_h2d_elp_ts)->
  
  u32 : ()->new Uint32Array @host.buffer
  i32 : ()->new  Int32Array @host.buffer
  u16 : ()->new Uint16Array @host.buffer
  i16 : ()->new  Int16Array @host.buffer
  u8  : ()->new Uint8Array  @host.buffer
  i8  : ()->new  Int8Array  @host.buffer
  

class @GPU_kernel
  parent_ctx  : null
  _kernel     : null
  
  warp_size   : 0
  worker_count: 0
  code        : ""
  
  last_elp_ts : -1
  
  arg_list_set : (arg_list)->
    k_idx = 0
    for arg in arg_list
      # TODO GPU_image buffer + size_x + size_y
      if arg instanceof Number
        gpu_wrapper.kernel_set_arg @_kernel, k_idx++, +arg, "float"
      else if "number" == typeof arg
        type = "int"
        if Math.round(arg) != arg
          throw new Error "float is not supported #{arg}. Use new Number(...)"
        if !isFinite arg
          throw new Error "NaN float is not supported #{arg}. Use new Number(...)"
        gpu_wrapper.kernel_set_arg @_kernel, k_idx++, arg, "int"
      else if arg instanceof module.GPU_buffer
        gpu_wrapper.kernel_set_arg @_kernel, k_idx++, arg._device_buf
      else
        perr arg
        throw new Error "bad arg type"
    # внутри foreach, не юзать
    # send_arg_list = []
    # for arg in arg_list
    #   # TODO GPU_image buffer + size_x + size_y
    #   if "number" == typeof arg
    #     type = "uint"
    #     type = "int" if arg < 0
    #     if Math.round(arg) != arg
    #       throw new Error "float is not supported #{arg}"
    #     if !isFinite arg
    #       throw new Error "float is not supported #{arg}"
    #     hash = {}
    #     hash[type] = arg
    #     send_arg_list.push hash
    #   else if arg instanceof module.GPU_buffer
    #     send_arg_list.push arg._device_buf
    #   else
    #     perr arg
    #     throw new Error "bad arg type"
    
    # gpu_wrapper.kernel_set_args @_kernel, send_arg_list
    return
  
  last_elp_ts : -1
  
  _bench : (@last_elp_ts)->
