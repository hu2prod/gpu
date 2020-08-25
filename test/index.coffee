assert = require "assert"

gpu_mod = require "../src/index.coffee"
now = require "performance-now"
bench_count     = 20
parallel_count  = 4 # 4 async CU
# parallel_count  = 1

default_buf_size    = 2*1024*1024
default_warp_size    = 128
default_worker_count = 512

describe "index section", ()->
  it "workflow", (on_end)->
    dev = gpu_mod.device_list_get()[0]
    ctx = dev.ctx()
    queue = ctx.queue()
    buf = ctx.buffer default_buf_size
    code = """
      __kernel void _main(
        __global       u32* res_buf
      ) {
        res_buf[0] += 1;
      }
      """
    await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
    kernel.arg_list_set [buf]
    
    elp_ts_list = []
    for i in [0 ... bench_count]
      buf.u32()[0] = 1
      
      start_ts = now()
      await queue.h2d buf, defer(err); return on_end err if err
      await queue.kernel kernel, defer(err); return on_end err if err
      await queue.d2h buf, defer(err); return on_end err if err
      await queue.finish defer(err); return on_end err if err # optional
      elp_ts_list.push now() - start_ts
      
      u32_view = buf.u32()
      assert.strictEqual u32_view[0], 2
      assert.strictEqual u32_view[1], 0
    
    elp_ts_list.sort (a,b)->a-b
    p "elp_ts ", elp_ts_list.map((t)->t.toFixed 2).join " "
    
    on_end null
    return
  
  it "workflow FASTER", (on_end)->
    dev = gpu_mod.device_list_get()[0]
    ctx = dev.ctx()
    queue = ctx.queue()
    queue.FASTER = true
    buf = ctx.buffer default_buf_size
    code = """
      __kernel void _main(
        __global       u32* res_buf
      ) {
        res_buf[0] += 1;
      }
      """
    await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
    kernel.arg_list_set [buf]
    
    elp_ts_list = []
    for i in [0 ... bench_count]
      buf.u32()[0] = 1
      
      start_ts = now()
      await queue.h2d buf, defer(err); return on_end err if err
      await queue.kernel kernel, defer(err); return on_end err if err
      await queue.d2h buf, defer(err); return on_end err if err
      await queue.finish defer(err); return on_end err if err
      elp_ts_list.push now() - start_ts
      
      u32_view = buf.u32()
      assert.strictEqual u32_view[0], 2
      assert.strictEqual u32_view[1], 0
    
    elp_ts_list.sort (a,b)->a-b
    p "elp_ts ", elp_ts_list.map((t)->t.toFixed 2).join " "
    
    on_end null
    return
  
  it "workflow FASTER + !FLUSH", (on_end)->
    dev = gpu_mod.device_list_get()[0]
    ctx = dev.ctx()
    queue = ctx.queue()
    queue.FASTER = true
    queue.FLUSH = false
    buf = ctx.buffer default_buf_size
    code = """
      __kernel void _main(
        __global       u32* res_buf
      ) {
        res_buf[0] += 1;
      }
      """
    await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
    kernel.arg_list_set [buf]
    
    elp_ts_list = []
    for i in [0 ... bench_count]
      buf.u32()[0] = 1
      
      start_ts = now()
      await queue.h2d buf, defer(err); return on_end err if err
      await queue.kernel kernel, defer(err); return on_end err if err
      await queue.d2h buf, defer(err); return on_end err if err
      await queue.finish defer(err); return on_end err if err
      elp_ts_list.push now() - start_ts
      
      u32_view = buf.u32()
      assert.strictEqual u32_view[0], 2
      assert.strictEqual u32_view[1], 0
    
    elp_ts_list.sort (a,b)->a-b
    p "elp_ts ", elp_ts_list.map((t)->t.toFixed 2).join " "
    
    on_end null
    return
  
  it "workflow chain parallel", (on_end)->
    dev = gpu_mod.device_list_get()[0]
    ctx = dev.ctx()
    base_queue = ctx.queue()
    
    queue_list = []
    buf_list = []
    kernel_list = []
    for i in [0 ... parallel_count]
      queue_list.push base_queue.fork()
      buf_list.push buf = ctx.buffer default_buf_size
      code = """
        __kernel void _main(
          __global       u32* res_buf
        ) {
          res_buf[0] += 1;
        }
        """
      await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
      kernel.arg_list_set [buf]
      kernel_list.push kernel
    
    elp_ts_list = []
    for j in [0 ... bench_count]
      for i in [0 ... parallel_count]
        buf = buf_list[i]
        u32_view = buf.u32()
        u32_view[0] = 1
      
      start_ts = now()
      for i in [0 ... parallel_count]
        buf = buf_list[i]
        kernel = kernel_list[i]
        queue = queue_list[i]
        await queue.h2d buf, defer(err); return on_end err if err
        await queue.kernel kernel, defer(err); return on_end err if err
        await queue.d2h buf, defer(err); return on_end err if err
      
      await
        for i in [0 ... parallel_count]
          cb = defer()
          do (i, cb)->
            buf = buf_list[i]
            queue = queue_list[i]
            await queue.finish defer(err); # return on_end err if err
            
            u32_view = buf.u32()
            assert.strictEqual u32_view[0], 2, "thread #{i}/#{parallel_count} failed"
            assert.strictEqual u32_view[1], 0, "thread #{i}/#{parallel_count} failed"
            cb()
        
      elp_ts_list.push now() - start_ts
    
    elp_ts_list.sort (a,b)->a-b
    p "NOTE parallel_count = #{parallel_count}"
    p "elp_ts ", elp_ts_list.map((t)->t.toFixed 2).join " "
    p "/count ", elp_ts_list.map((t)->(t/parallel_count).toFixed 2).join " "
    
    on_end null
    return
  
  it "workflow chain parallel + !FLUSH", (on_end)->
    dev = gpu_mod.device_list_get()[0]
    ctx = dev.ctx()
    base_queue = ctx.queue()
    base_queue.FLUSH = false
    
    queue_list = []
    buf_list = []
    kernel_list = []
    for i in [0 ... parallel_count]
      queue_list.push base_queue.fork()
      buf_list.push buf = ctx.buffer default_buf_size
      code = """
        __kernel void _main(
          __global       u32* res_buf
        ) {
          res_buf[0] += 1;
        }
        """
      await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
      kernel.arg_list_set [buf]
      kernel_list.push kernel
    
    elp_ts_list = []
    for j in [0 ... bench_count]
      for i in [0 ... parallel_count]
        buf = buf_list[i]
        u32_view = buf.u32()
        u32_view[0] = 1
      
      start_ts = now()
      for i in [0 ... parallel_count]
        buf = buf_list[i]
        kernel = kernel_list[i]
        queue = queue_list[i]
        await queue.h2d buf, defer(err); return on_end err if err
        await queue.kernel kernel, defer(err); return on_end err if err
        await queue.d2h buf, defer(err); return on_end err if err
      
      await
        for i in [0 ... parallel_count]
          cb = defer()
          do (i, cb)->
            buf = buf_list[i]
            queue = queue_list[i]
            await queue.finish defer(err); # return on_end err if err
            
            u32_view = buf.u32()
            assert.strictEqual u32_view[0], 2, "thread #{i}/#{parallel_count} failed"
            assert.strictEqual u32_view[1], 0, "thread #{i}/#{parallel_count} failed"
            cb()
        
      elp_ts_list.push now() - start_ts
    
    elp_ts_list.sort (a,b)->a-b
    p "NOTE parallel_count = #{parallel_count}"
    p "elp_ts ", elp_ts_list.map((t)->t.toFixed 2).join " "
    p "/count ", elp_ts_list.map((t)->(t/parallel_count).toFixed 2).join " "
    
    on_end null
    return
  
  
  
  it "workflow chain parallel + FASTER", (on_end)->
    dev = gpu_mod.device_list_get()[0]
    ctx = dev.ctx()
    base_queue = ctx.queue()
    base_queue.FASTER = true
    
    queue_list = []
    buf_list = []
    kernel_list = []
    for i in [0 ... parallel_count]
      queue_list.push base_queue.fork()
      buf_list.push buf = ctx.buffer default_buf_size
      code = """
        __kernel void _main(
          __global       u32* res_buf
        ) {
          res_buf[0] += 1;
        }
        """
      await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
      kernel.arg_list_set [buf]
      kernel_list.push kernel
    
    elp_ts_list = []
    for j in [0 ... bench_count]
      for i in [0 ... parallel_count]
        buf = buf_list[i]
        u32_view = buf.u32()
        u32_view[0] = 1
      
      start_ts = now()
      for i in [0 ... parallel_count]
        buf = buf_list[i]
        kernel = kernel_list[i]
        queue = queue_list[i]
        await queue.h2d buf, defer(err); return on_end err if err
        await queue.kernel kernel, defer(err); return on_end err if err
        await queue.d2h buf, defer(err); return on_end err if err
      
      await
        for i in [0 ... parallel_count]
          cb = defer()
          do (i, cb)->
            buf = buf_list[i]
            queue = queue_list[i]
            await queue.finish defer(err); # return on_end err if err
            
            u32_view = buf.u32()
            assert.strictEqual u32_view[0], 2, "thread #{i}/#{parallel_count} failed"
            assert.strictEqual u32_view[1], 0, "thread #{i}/#{parallel_count} failed"
            cb()
        
      elp_ts_list.push now() - start_ts
    
    elp_ts_list.sort (a,b)->a-b
    p "NOTE parallel_count = #{parallel_count}"
    p "elp_ts ", elp_ts_list.map((t)->t.toFixed 2).join " "
    p "/count ", elp_ts_list.map((t)->(t/parallel_count).toFixed 2).join " "
    
    on_end null
    return
  
  it "workflow chain parallel + FASTER + !FLUSH", (on_end)->
    dev = gpu_mod.device_list_get()[0]
    ctx = dev.ctx()
    base_queue = ctx.queue()
    base_queue.FASTER = true
    base_queue.FLUSH = false
    
    queue_list = []
    buf_list = []
    kernel_list = []
    for i in [0 ... parallel_count]
      queue_list.push base_queue.fork()
      buf_list.push buf = ctx.buffer default_buf_size
      code = """
        __kernel void _main(
          __global       u32* res_buf
        ) {
          res_buf[0] += 1;
        }
        """
      await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
      kernel.arg_list_set [buf]
      kernel_list.push kernel
    
    elp_ts_list = []
    for j in [0 ... bench_count]
      for i in [0 ... parallel_count]
        buf = buf_list[i]
        u32_view = buf.u32()
        u32_view[0] = 1
      
      start_ts = now()
      for i in [0 ... parallel_count]
        buf = buf_list[i]
        kernel = kernel_list[i]
        queue = queue_list[i]
        await queue.h2d buf, defer(err); return on_end err if err
        await queue.kernel kernel, defer(err); return on_end err if err
        await queue.d2h buf, defer(err); return on_end err if err
      
      await
        for i in [0 ... parallel_count]
          cb = defer()
          do (i, cb)->
            buf = buf_list[i]
            queue = queue_list[i]
            await queue.finish defer(err); # return on_end err if err
            
            u32_view = buf.u32()
            assert.strictEqual u32_view[0], 2, "thread #{i}/#{parallel_count} failed"
            assert.strictEqual u32_view[1], 0, "thread #{i}/#{parallel_count} failed"
            cb()
        
      elp_ts_list.push now() - start_ts
    
    elp_ts_list.sort (a,b)->a-b
    p "NOTE parallel_count = #{parallel_count}"
    p "elp_ts ", elp_ts_list.map((t)->t.toFixed 2).join " "
    p "/count ", elp_ts_list.map((t)->(t/parallel_count).toFixed 2).join " "
    
    on_end null
    return
  
  describe "out of order (ooo)", ()->
    it "workflow ooo chain", (on_end)->
      dev = gpu_mod.device_list_get()[0]
      ctx = dev.ctx()
      queue = ctx.queue_ooo()
      buf = ctx.buffer default_buf_size
      code = """
        __kernel void _main(
          __global       u32* res_buf
        ) {
          res_buf[0] += 1;
        }
        """
      await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
      kernel.arg_list_set [buf]
      
      elp_ts_list = []
      for i in [0 ... bench_count]
        u32_view = buf.u32()
        u32_view[0] = 1
        
        start_ts = now()
        await queue.h2d buf, defer(err); return on_end err if err
        await queue.kernel kernel, defer(err); return on_end err if err
        await queue.d2h buf, defer(err); return on_end err if err
        await queue.finish defer(err); return on_end err if err
        elp_ts_list.push now() - start_ts
        
        u32_view = buf.u32()
        assert.strictEqual u32_view[0], 2
        assert.strictEqual u32_view[1], 0
      
      elp_ts_list.sort (a,b)->a-b
      p "elp_ts ", elp_ts_list.map((t)->t.toFixed 2).join " "
      
      on_end null
      return
    
    it "workflow ooo chain + !FLUSH", (on_end)->
      dev = gpu_mod.device_list_get()[0]
      ctx = dev.ctx()
      queue = ctx.queue_ooo()
      queue.FLUSH = false
      buf = ctx.buffer default_buf_size
      code = """
        __kernel void _main(
          __global       u32* res_buf
        ) {
          res_buf[0] += 1;
        }
        """
      await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
      kernel.arg_list_set [buf]
      
      elp_ts_list = []
      for i in [0 ... bench_count]
        u32_view = buf.u32()
        u32_view[0] = 1
        
        start_ts = now()
        await queue.h2d buf, defer(err); return on_end err if err
        await queue.kernel kernel, defer(err); return on_end err if err
        await queue.d2h buf, defer(err); return on_end err if err
        await queue.finish defer(err); return on_end err if err
        elp_ts_list.push now() - start_ts
        
        u32_view = buf.u32()
        assert.strictEqual u32_view[0], 2
        assert.strictEqual u32_view[1], 0
      
      elp_ts_list.sort (a,b)->a-b
      p "elp_ts ", elp_ts_list.map((t)->t.toFixed 2).join " "
      
      on_end null
      return
    
    it "workflow ooo chain parallel", (on_end)->
      dev = gpu_mod.device_list_get()[0]
      ctx = dev.ctx()
      base_queue = ctx.queue_ooo()
      
      queue_list = []
      buf_list = []
      kernel_list = []
      for i in [0 ... parallel_count]
        queue_list.push base_queue.fork()
        buf_list.push buf = ctx.buffer default_buf_size
        code = """
          __kernel void _main(
            __global       u32* res_buf
          ) {
            res_buf[0] += 1;
          }
          """
        await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
        kernel.arg_list_set [buf]
        kernel_list.push kernel
      
      elp_ts_list = []
      for j in [0 ... bench_count]
        for i in [0 ... parallel_count]
          buf = buf_list[i]
          u32_view = buf.u32()
          u32_view[0] = 1
        
        start_ts = now()
        for i in [0 ... parallel_count]
          buf = buf_list[i]
          kernel = kernel_list[i]
          queue = queue_list[i]
          await queue.h2d buf, defer(err); return on_end err if err
          await queue.kernel kernel, defer(err); return on_end err if err
          await queue.d2h buf, defer(err); return on_end err if err
        
        await
          for i in [0 ... parallel_count]
            cb = defer()
            do (i, cb)->
              buf = buf_list[i]
              queue = queue_list[i]
              await queue.finish defer(err); # return on_end err if err
              
              u32_view = buf.u32()
              assert.strictEqual u32_view[0], 2, "thread #{i}/#{parallel_count} failed"
              assert.strictEqual u32_view[1], 0, "thread #{i}/#{parallel_count} failed"
              cb()
          
        elp_ts_list.push now() - start_ts
      
      elp_ts_list.sort (a,b)->a-b
      p "NOTE parallel_count = #{parallel_count}"
      p "elp_ts ", elp_ts_list.map((t)->t.toFixed 2).join " "
      p "/count ", elp_ts_list.map((t)->(t/parallel_count).toFixed 2).join " "
      
      on_end null
      return
    
    # Прим. FLUSH выключен на уровне nooocl_wrapper
    # Так что код выполняется тупо идентичный
    # Но смешно смотреть на разные результаты
    it "workflow ooo chain parallel + !FLUSH", (on_end)->
      dev = gpu_mod.device_list_get()[0]
      ctx = dev.ctx()
      base_queue = ctx.queue_ooo()
      base_queue.FLUSH = false
      
      queue_list = []
      buf_list = []
      kernel_list = []
      for i in [0 ... parallel_count]
        queue_list.push base_queue.fork()
        buf_list.push buf = ctx.buffer default_buf_size
        code = """
          __kernel void _main(
            __global       u32* res_buf
          ) {
            res_buf[0] += 1;
          }
          """
        await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
        kernel.arg_list_set [buf]
        kernel_list.push kernel
      
      elp_ts_list = []
      for j in [0 ... bench_count]
        for i in [0 ... parallel_count]
          buf = buf_list[i]
          u32_view = buf.u32()
          u32_view[0] = 1
        
        start_ts = now()
        for i in [0 ... parallel_count]
          buf = buf_list[i]
          kernel = kernel_list[i]
          queue = queue_list[i]
          await queue.h2d buf, defer(err); return on_end err if err
          await queue.kernel kernel, defer(err); return on_end err if err
          await queue.d2h buf, defer(err); return on_end err if err
        
        await
          for i in [0 ... parallel_count]
            cb = defer()
            do (i, cb)->
              buf = buf_list[i]
              queue = queue_list[i]
              await queue.finish defer(err); # return on_end err if err
              
              u32_view = buf.u32()
              assert.strictEqual u32_view[0], 2, "thread #{i}/#{parallel_count} failed"
              assert.strictEqual u32_view[1], 0, "thread #{i}/#{parallel_count} failed"
              cb()
          
        elp_ts_list.push now() - start_ts
      
      elp_ts_list.sort (a,b)->a-b
      p "NOTE parallel_count = #{parallel_count}"
      p "elp_ts ", elp_ts_list.map((t)->t.toFixed 2).join " "
      p "/count ", elp_ts_list.map((t)->(t/parallel_count).toFixed 2).join " "
      
      on_end null
      return
  
  describe "buffer", ()->
    it "buffer_host", (on_end)->
      dev = gpu_mod.device_list_get()[0]
      ctx = dev.ctx()
      queue = ctx.queue()
      buf = ctx.buffer_host 100
      buf.u32()[0] = 1
      code = """
        __kernel void _main(
          __global       u32* res_buf
        ) {
          res_buf[0] += 1;
        }
        """
      await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
      kernel.arg_list_set [buf]
      
      # await queue.h2d buf, defer(err); return on_end err if err
      await queue.kernel kernel, defer(err); return on_end err if err
      # await queue.d2h buf, defer(err); return on_end err if err
      
      u32_view = buf.u32()
      assert.strictEqual u32_view[0], 2
      assert.strictEqual u32_view[1], 0
      
      on_end null
      return
    
    it "buffer_host + h2d + d2h", (on_end)->
      dev = gpu_mod.device_list_get()[0]
      ctx = dev.ctx()
      queue = ctx.queue()
      buf = ctx.buffer_host 100
      buf.u32()[0] = 1
      code = """
        __kernel void _main(
          __global       u32* res_buf
        ) {
          res_buf[0] += 1;
        }
        """
      await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
      kernel.arg_list_set [buf]
      
      await queue.h2d buf, defer(err); return on_end err if err
      await queue.kernel kernel, defer(err); return on_end err if err
      await queue.d2h buf, defer(err); return on_end err if err
      
      u32_view = buf.u32()
      assert.strictEqual u32_view[0], 2
      assert.strictEqual u32_view[1], 0
      
      on_end null
      return
    
    it "buffer_device", (on_end)->
      dev = gpu_mod.device_list_get()[0]
      ctx = dev.ctx()
      buf = ctx.buffer_device 100
      
      on_end null
      return
    
    it "buffer typed access", (on_end)->
      dev = gpu_mod.device_list_get()[0]
      ctx = dev.ctx()
      buf = ctx.buffer 100
      i32 = buf.i32()
      i32[0] = -2
      u32 = buf.u32()
      assert.strictEqual u32[0], 2**32 - 2
      u16 = buf.u16()
      assert.strictEqual u16[0], 2**16 - 2
      assert.strictEqual u16[1], 2**16 - 1
      i16 = buf.i16()
      assert.strictEqual i16[0], -2
      assert.strictEqual i16[1], -1
      u8 = buf.u8()
      assert.strictEqual u8[0], 2**8 - 2
      assert.strictEqual u8[1], 2**8 - 1
      assert.strictEqual u8[2], 2**8 - 1
      assert.strictEqual u8[3], 2**8 - 1
      i8 = buf.i8()
      assert.strictEqual i8[0], -2
      assert.strictEqual i8[1], -1
      assert.strictEqual i8[2], -1
      assert.strictEqual i8[3], -1
      
      on_end null
      return
  
  describe "kernel", ()->
    it "kernel_path", (on_end)->
      dev = gpu_mod.device_list_get()[0]
      ctx = dev.ctx()
      await ctx.kernel_path __dirname+"/kernel.cl", 128, 1024, defer(err, kernel); return on_end err if err
      
      on_end null
      return
    
    it "kernel const arg", (on_end)->
      dev = gpu_mod.device_list_get()[0]
      ctx = dev.ctx()
      queue = ctx.queue()
      buf = ctx.buffer default_buf_size
      code = """
        __kernel void _main(
          __global       u32* res_buf,
              const u32  thread_count
        ) {
          res_buf[0] = 1;
        }
        """
      await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
      kernel.arg_list_set [
        buf
        1
      ]
      await queue.h2d buf, defer(err); return on_end err if err
      await queue.kernel kernel, defer(err); return on_end err if err
      await queue.d2h buf, defer(err); return on_end err if err
      
      on_end null
      return
    
    it "kernel const arg neg", (on_end)->
      dev = gpu_mod.device_list_get()[0]
      ctx = dev.ctx()
      queue = ctx.queue()
      buf = ctx.buffer default_buf_size
      code = """
        __kernel void _main(
          __global       u32* res_buf,
              const i32  thread_count
        ) {
          res_buf[0] = 1;
        }
        """
      await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
      kernel.arg_list_set [
        buf
        -1
      ]
      await queue.h2d buf, defer(err); return on_end err if err
      await queue.kernel kernel, defer(err); return on_end err if err
      await queue.d2h buf, defer(err); return on_end err if err
      
      on_end null
      return
  
  describe "throws", ()->
    describe "buffer", ()->
      it "buffer no size", ()->
        dev = gpu_mod.device_list_get()[0]
        ctx = dev.ctx()
        assert.throws ()->
          buf = ctx.buffer()
        
        return
      
      it "buffer negative size", ()->
        dev = gpu_mod.device_list_get()[0]
        ctx = dev.ctx()
        assert.throws ()->
          buf = ctx.buffer -1
        
        return
      
      it "buffer_device d2h h2d", (on_end)->
        dev = gpu_mod.device_list_get()[0]
        ctx = dev.ctx()
        queue = ctx.queue()
        buf = ctx.buffer_device 100
        
        await queue.h2d buf, defer(err); return on_end new Error "expected error" if !err
        await queue.d2h buf, defer(err); return on_end new Error "expected error" if !err
        
        on_end null
        return
    
    describe "kernel", ()->
      it "bad kernel_path", (on_end)->
        dev = gpu_mod.device_list_get()[0]
        ctx = dev.ctx()
        await ctx.kernel_path __dirname+"/wtf", 128, 1024, defer(err, kernel); return on_end new Error "expected error" if !err
        
        on_end null
        return
      
      it "bad kernel code", (on_end)->
        dev = gpu_mod.device_list_get()[0]
        ctx = dev.ctx()
        buf = ctx.buffer default_buf_size
        
        code = """
          __kernel void _main(
            __global       u32* res_buf
          ) {
            res_buf[0] = 1
          }
          """
        await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err)
        if !err
          return on_end new Error "expected error"
        
        on_end null
        return
      
      it "bad launch size", (on_end)->
        dev = gpu_mod.device_list_get()[0]
        ctx = dev.ctx()
        buf = ctx.buffer default_buf_size
        code = """
          __kernel void _main(
            __global       u32* res_buf
          ) {
            res_buf[0] = 1;
          }
          """
        await ctx.kernel_code code, 128, -1,  defer(err); return on_end new Error "expected error" if !err
        await ctx.kernel_code code, -1, 1024, defer(err); return on_end new Error "expected error" if !err
        await ctx.kernel_code code, 128, NaN,  defer(err); return on_end new Error "expected error" if !err
        await ctx.kernel_code code, NaN, 1024, defer(err); return on_end new Error "expected error" if !err
        inf = Infinity
        await ctx.kernel_code code, 128, inf,  defer(err); return on_end new Error "expected error" if !err
        await ctx.kernel_code code, inf, 1024, defer(err); return on_end new Error "expected error" if !err
        inf = -Infinity
        await ctx.kernel_code code, 128, inf,  defer(err); return on_end new Error "expected error" if !err
        await ctx.kernel_code code, inf, 1024, defer(err); return on_end new Error "expected error" if !err
          
        
        on_end null
        return
      
      # Fails (
      # it "kernel missing arg", (on_end)->
      #   dev = gpu_mod.device_list_get()[0]
      #   ctx = dev.ctx()
      #   queue = ctx.queue()
      #   buf = ctx.buffer default_buf_size
      #   code = """
      #     __kernel void _main(
      #       __global       u32* res_buf,
      #           const i32  thread_count
      #     ) {
      #       res_buf[0] = 1;
      #     }
      #     """
      #   await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
      #   assert.throws ()->
      #     kernel.arg_list_set [
      #       buf
      #     ]
      #   
      #   on_end null
      #   return
      
      it "kernel bad arg type", (on_end)->
        dev = gpu_mod.device_list_get()[0]
        ctx = dev.ctx()
        queue = ctx.queue()
        buf = ctx.buffer default_buf_size
        code = """
          __kernel void _main(
            __global       u32* res_buf,
                const i32  thread_count
          ) {
            res_buf[0] = 1;
          }
          """
        await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
        assert.throws ()->
          kernel.arg_list_set [
            buf
            buf
          ]
        
        on_end null
        return
      
      # it "workflow FASTER but no finish but works (can be potentially unstable)", (on_end)->
      #   dev = gpu_mod.device_list_get()[0]
      #   ctx = dev.ctx()
      #   queue = ctx.queue()
      #   queue.FASTER = true
      #   # buf = ctx.buffer default_buf_size
      #   buf = ctx.buffer 100
      #   u32_view = buf.u32()
      #   u32_view[0] = 1
      #   code = """
      #     __kernel void _main(
      #       __global       u32* res_buf
      #     ) {
      #       res_buf[0] += 1;
      #     }
      #     """
      #   # await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
      #   await ctx.kernel_code code, 128, 1024, defer(err, kernel); return on_end err if err
      #   kernel.arg_list_set [buf]
      #   
      #   # 1 ms, but that's enough to d2h
      #   await queue.h2d buf, defer(err); return on_end err if err
      #   await queue.kernel kernel, defer(err); return on_end err if err
      #   await queue.d2h buf, defer(err); return on_end err if err
      #   # await queue.finish defer(err); return on_end err if err
      #   
      #   u32_view = buf.u32()
      #   assert.strictEqual u32_view[0], 2
      #   
      #   on_end null
      #   return
      
      it "workflow FASTER but no finish (can be potentially unstable)", (on_end)->
        dev = gpu_mod.device_list_get()[0]
        ctx = dev.ctx()
        queue = ctx.queue()
        queue.FASTER = true
        # buf = ctx.buffer default_buf_size
        buf = ctx.buffer 10*1024*1024 # we need big buffer for really slow copy
        u32_view = buf.u32()
        u32_view[0] = 1
        code = """
          __kernel void _main(
            __global       u32* res_buf
          ) {
            res_buf[0] += 1;
          }
          """
        await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
        kernel.arg_list_set [buf]
        
        # 31 ms, but without finish will fail
        await queue.h2d buf, defer(err); return on_end err if err
        await queue.kernel kernel, defer(err); return on_end err if err
        await queue.d2h buf, defer(err); return on_end err if err
        # await queue.finish defer(err); return on_end err if err
        
        u32_view = buf.u32()
        assert.throws ()->
          assert.strictEqual u32_view[0], 2
        
        on_end null
        return
    